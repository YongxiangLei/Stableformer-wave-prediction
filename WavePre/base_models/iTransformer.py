import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class Model(nn.Module):
    """
    iTransformer model adapted for multi-step forecasting.
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.his_len
        self.pred_len = configs.pred_len

        # Embedding for input data
        self.enc_embedding = DataEmbedding_inverted(configs.his_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Projection layer for forecasting
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for multi-step forecasting.
        Args:
            x_enc: Input sequence for encoding (shape: [batch_size, seq_len, num_features])
            x_mark_enc: Time features for encoding (shape: [batch_size, seq_len, num_features])
            x_dec: Input sequence for decoding (optional, shape: [batch_size, pred_len, num_features])
            x_mark_dec: Time features for decoding (optional, shape: [batch_size, pred_len, num_features])
            mask: Attention mask (optional)
        
        Returns:
            dec_out: Forecasted output (shape: [batch_size, pred_len, num_features])
        """
        # Normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Project encoder output to future steps
        # We assume the output shape after encoder is [batch_size, seq_len, d_model]
        dec_out = self.projection(enc_out)[:, -self.pred_len:, :]

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

