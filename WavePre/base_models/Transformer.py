import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=configs.d_model, nhead=configs.n_heads, dim_feedforward=configs.d_ff, dropout=configs.dropout),
            num_layers=configs.e_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=configs.d_model, nhead=configs.n_heads, dim_feedforward=configs.d_ff, dropout=configs.dropout),
            num_layers=configs.d_layers
        )
        
        self.input_layer = nn.Linear(configs.enc_in, configs.d_model)
        self.output_layer = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, src, tgt, dec_inp, src_mask=None, tgt_mask=None):
        # src: (batch_size, sequence_length, enc_in)
        # dec_inp: (batch_size, pred_len, dec_in)
        
        # Prepare input for the encoder
        src = self.input_layer(src)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model) for Transformer
        
        # Encoder
        memory = self.encoder(src, src_mask)  # (seq_len, batch_size, d_model)
        
        # Prepare input for the decoder
        dec_inp = self.input_layer(dec_inp)  # (batch_size, pred_len, d_model)
        dec_inp = dec_inp.permute(1, 0, 2)  # (pred_len, batch_size, d_model)

        # Decoder
        output = self.decoder(dec_inp, memory, tgt_mask)  # (pred_len, batch_size, d_model)
        
        # Prepare output
        output = self.output_layer(output.permute(1, 0, 2))  # (batch_size, pred_len, c_out)
        
        return output
