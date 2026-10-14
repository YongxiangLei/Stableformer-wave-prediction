import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding

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

        # Encoder setup
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

        # Decoder setup
        self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for multi-step forecasting.
        Args:
            x_enc: Input sequence for encoding (shape: [batch_size, seq_len, num_features])
            x_mark_enc: Time features for encoding (shape: [batch_size, seq_len, num_features])
            x_dec: Input sequence for decoding (shape: [batch_size, pred_len, num_features])
            x_mark_dec: Time features for decoding (shape: [batch_size, pred_len, num_features])
        
        Returns:
            dec_out: Forecasted output (shape: [batch_size, pred_len, num_features])
        """
        # Normalize encoder input
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Encode the input sequence
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Prepare decoder input (shifted target sequence)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # Return the predicted sequence for the next 'pred_len' steps
        dec_out = dec_out[:, -self.pred_len:, :]

        # De-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def predict_future(self, x_enc, x_mark_enc, x_dec_start, x_mark_dec_start, future_steps):
        """
        Generate multi-step future predictions using a recursive decoder.
        Args:
            x_enc: past input sequence
            x_mark_enc: time features for encoding
            x_dec_start: start of the decoder sequence (usually last value of encoder input)
            x_mark_dec_start: time features for decoding
            future_steps: number of steps to predict into the future
        """
        # Encoding the past inputs
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        preds = []
        dec_inp = x_dec_start

        for _ in range(future_steps):
            # Decoding to predict the next step
            dec_out = self.dec_embedding(dec_inp, x_mark_dec_start)
            dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

            # Extract the predicted step and append to predictions
            next_pred = dec_out[:, -1:, :]  # We only care about the last time step prediction
            preds.append(next_pred)

            # Update decoder input for next step prediction
            dec_inp = torch.cat([dec_inp[:, 1:, :], next_pred], dim=1)  # Slide window forward

        # Concatenate all predicted steps
        preds = torch.cat(preds, dim=1)
        return preds
