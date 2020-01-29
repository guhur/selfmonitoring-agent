import torch
import torch.nn as nn

from models.modules import (
    build_mlp,
    SoftAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    create_mask,
    proj_masking,
    PositionalEncoding,
)
from models.film import FiLMGenerator, FiLMedResBlocks, FiLMTail


class SelfMonitoring(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(
        self,
        opts,
        img_fc_dim,
        img_fc_use_batchnorm,
        img_dropout,
        img_feat_input_dim,
        rnn_hidden_size,
        rnn_dropout,
        max_len,
        film_size=2048,
        fc_bias=True,
        max_navigable=16,
        conv_hidden=2048,
        num_resblocks=8,
    ):
        super(SelfMonitoring, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = {
            "input_dim": img_feat_input_dim,
            "hidden_dims": img_fc_dim,
            "use_batchnorm": img_fc_use_batchnorm,
            "dropout": img_dropout,
            "fc_bias": fc_bias,
            "relu": opts.mlp_relu,
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(
            img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size
        )

        self.lang_position = PositionalEncoding(
            rnn_hidden_size, dropout=0.1, max_len=max_len
        )

        self.logit_fc = nn.Linear(film_size, img_fc_dim[-1])

        self.h2_fc_lstm = nn.Linear(
            rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias
        )

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1), nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1), nn.Tanh()
            )

        self.num_predefined_action = 1

        # EDIT: add FiLM
        self.resnet = torch.hub.load("pytorch/vision", "resnet152", pretrained=True)
        self.resnet = nn.Sequential(self.resnet.modules()[:-2])

        self.film_gen = FiLMGenerator(
            context_size=rnn_hidden_size,
            num_resblocks=num_resblocks,
            conv_hidden=conv_hidden,
        )

        self.film = FiLMedResBlocks(
            num_blocks=num_resblocks,
            conv_hidden=conv_hidden,
            with_batch_norm=True,
        )

        self.film_tail = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        nav_imgs,
        navigable_ang_feat,
        pre_feat,
        question,
        h_0,
        c_0,
        ctx,
        pre_ctx_attend,
        navigable_index=None,
        ctx_mask=None,
    ):
        """ Takes a single step in the decoder

        navigable_feat: batch x max_navigable x feature_size
        nav_img: batch x max_navigable x 3 x H x W

        pre_feat: previous attended feature, batch x feature_size

        question: this should be a single vector representing instruction

        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size = nav_imgs.shape[0]

        index_length = [
            len(_index) + self.num_predefined_action
            for _index in navigable_index
        ]
        navigable_mask = create_mask(
            batch_size, self.max_navigable, index_length
        )

        # Get nav_feats from FiLM
        nav_imgs = self.resnet(nav_imgs)
        beta_gamma = self.film_gen(h_0)
        nav_imgs = self.film(nav_imgs, beta_gamma)
        navigable_feat = nav_imgs.mean(-1).mean(-1)
        navigable_feat = torch.cat([
            navigable_feat,
            navigable_ang_feat],
            2
        )

        proj_navigable_feat = proj_masking(
            navigable_feat, self.proj_navigable_mlp, navigable_mask
        )
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(
            self.h1_fc(h_0), positioned_ctx, mask=ctx_mask
        )

        weighted_img_feat, img_attn = self.soft_attn(
            self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask
        )

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat(
            (proj_pre_feat, weighted_img_feat, weighted_ctx), 1
        )

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(
            torch.cat((h_0, weighted_img_feat), 1)
        )

        h_1_value = self.dropout(
            torch.sigmoid(concat_value_input) * torch.tanh(c_1)
        )

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

        return (
            h_1,
            c_1,
            weighted_ctx,
            img_attn,
            ctx_attn,
            logit,
            value,
            navigable_mask,
            navigable_feat,
        )
