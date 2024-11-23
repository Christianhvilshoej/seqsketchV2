import torch
import torch.nn as nn

class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, ff_dim=256, num_layers=3):
        super().__init__()
        self.output_dim = 2  # For (x, y) coordinates

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.combined_embedding = nn.Linear(2 * d_model, d_model) 
        # Linear projection to noise prediction
        self.final = nn.Linear(d_model, self.output_dim)

    def forward(self, x_emb, c_emb, t_emb):
        """
        Args:
            x_emb: Target embedding (B, L, D) - noisy data.
            c_emb: Condition embedding (B, M, D) - clean historical data. Can be None.
            t_emb: Time embedding (B, D) - time encoding.

        Returns:
            prediction: (B, L, 2) [either noise or x0]
        """
        
        if c_emb is None:
            # If no condition is provided, only use time embedding
            t_expanded = t_emb.unsqueeze(1)  # Shape: (B, 1, D)
            combined_context = self.combined_embedding(torch.cat([t_expanded, t_expanded], dim=-1))  # Shape: (B, 1, D)
        else:
            _, M, _ = c_emb.size()
            t_expanded = t_emb.unsqueeze(1).repeat(1, M, 1)  # Shape: (B, M, D)
            context = torch.cat([c_emb, t_expanded], dim=-1)  # Shape: (B, M, 2 * D)
            combined_context = self.combined_embedding(context)  # Shape: (B, M, D)
        
        # Decode using the context
        decoded = self.decoder(x_emb, combined_context)  # (B, L, D)
        pred = self.final(decoded)  # (B, L, 2)
        return pred
