import torch.nn as nn
from seqsketch.models import TimeEmbedding
from seqsketch.models.encoder import StrokeEncoder
from seqsketch.models.decoder import CrossAttentionDecoder

class StrokeDenoiser(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.max_seq_length = params.max_seq_length
        self.max_strokes = params.max_strokes
        self.d_model = params.d_model
        self.mode = params.mode
        self.nhead = params.nhead
        self.num_encoder_layers = params.num_encoder_layers
        #self.dim_feedforward = params.dim_feedforward
        self.num_decoder_layers = params.num_decoder_layers
        #self.max_len = params.max_len
        self.time_embedding = TimeEmbedding(self.d_model)
        self.combined_embedding = nn.Linear(2 * self.d_model, self.d_model)
        self.encoder = StrokeEncoder(input_dim = 2,
                                     d_model = self.d_model,
                                     mode = self.mode,
                                     nhead = self.nhead,
                                     num_layers=self.num_encoder_layers)
        self.decoder = CrossAttentionDecoder(d_model=self.d_model, 
                                             nhead=self.nhead, 
                                             ff_dim=256, 
                                             num_layers=self.num_decoder_layers)
    def forward(self, x, t, c, x_mask = None, c_mask = None):
        """
        Args:
            x: Target stroke (B, L, 2) - noisy data.
            c: Condition strokes (B, N, L, 2) - clean historical data. Can be None.
            t: Timestep (B) 

        Returns:
            prediction: (B, L, 2) [either noise or x0]
        """
        # Encoder
        x_emb = self.encoder(x,x_mask)
        if c_mask.sum() == 0:
            # Whole batch is first stroke
            c_emb = None
        else:            
            c_emb = self.encoder(c,c_mask)
        t_emb = self.time_embedding(t)
        # Decoder
        pred = self.decoder(x_emb,c_emb,t_emb)
        return pred.unsqueeze(1) # B x 1 x L x 2
