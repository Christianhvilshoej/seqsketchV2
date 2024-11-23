import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class SingleStrokeEncoder(nn.Module):
    def __init__(self, input_dim=2, d_model=128, max_length=50,
                 positional_econding = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = d_model
        self.rnn = nn.GRU(input_dim, d_model, batch_first=True)
        self.pe = positional_econding
        # Positional Encoding
        #if self.pe:
        #    self.register_buffer("positional_encoding", self._generate_positional_encoding(d_model, max_length))

    def forward(self, stroke, mask=None):
        B, N, L, _ = stroke.size()
        # Reshape condition for individual sequence encoding
        stroke = stroke.view(B * N, L, -1)  # (B*N, L, 2)

        if mask is None:
            lengths = torch.tensor([L] * B * N)
        else:
            if len(mask.shape) == 4:
                mask = mask.sum(dim=-1) / 2  # (B x N x L)
            stroke_mask = mask.view(B * N, L)  # (B*N, L)
            lengths = stroke_mask.sum(dim=-1).cpu().long()  # (B*N,)

        # Identify valid sequences (non-zero lengths)
        valid_indices = lengths > 0
        valid_strokes = stroke[valid_indices]  # (num_valid, L, 2)
        valid_lengths = lengths[valid_indices]  # (num_valid,)
        # Encode only valid sequences
        if valid_strokes.size(0) > 0:  # Check if there are valid sequences
            packed_input = pack_padded_sequence(valid_strokes, valid_lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.rnn(packed_input)
            encoded, _ = pad_packed_sequence(packed_output, total_length=L, batch_first=True)  # (num_valid, L, hidden_dim)
        else:
            encoded = torch.zeros(B * N, L, self.hidden_dim, device=stroke.device)  # Handle no valid sequences

        # Create a zero tensor for all sequences and insert valid embeddings
        embeds = torch.zeros(B * N, L, self.hidden_dim, device=stroke.device)  # (B*N, L, hidden_dim)
        embeds[valid_indices] = encoded

        #if self.pe:
        #    # Add positional encoding
        #    embeds += self.positional_encoding[:L].unsqueeze(0).to(embeds.device)  # (1, L, hidden_dim)

        # Reshape back to (B, N, hidden_dim)
        embeds = embeds.view(B, N, L, self.hidden_dim)  # (B, N, L, hidden_dim)
        return embeds

    @staticmethod
    def _generate_positional_encoding(d_model, max_length):
        """
        Generate a sinusoidal positional encoding matrix.
        """
        position = torch.arange(max_length).unsqueeze(1)  # (max_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model / 2)
        pe = torch.zeros(max_length, d_model)  # (max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class MultiStrokeEncoder(nn.Module):
    def __init__(self, input_dim=128, num_filters=64, kernel_size=3, stride=2, pool_stride=2,
                 nhead = 4, num_layers=3, mode = "tf"):
        super().__init__()

        # Convolutional Layers with pooling after each convolution
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.pool1 = nn.MaxPool1d(pool_stride)  # Pooling after first convolution
        
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.pool2 = nn.MaxPool1d(pool_stride)  # Pooling after second convolution
        
        # Transformer Encoder
        self.mode = mode
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=num_filters*2, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.rnn = nn.GRU(input_dim, input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, stroke):
        # Apply first convolution and pooling
        stroke = stroke.permute(0, 2, 1)  # Change to (B, D, L) for Conv1d
        x = self.conv1(stroke)  # Output shape: (B, num_filters, L/stride)
        x = self.pool1(x)  # Output shape: (B, num_filters, (L/stride)/pool_stride)
        # Apply second convolution and pooling
        x = self.conv2(x)  # Output shape: (B, num_filters*2, (L/stride^2))
        x = self.pool2(x)  # Output shape: (B, num_filters*2, (L/stride^2)/pool_stride)
        # Reshape for transformer (batch_size, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # Output shape: (B, seq_output_length, num_filters*2)
        # Pass through final layer
        if self.mode == "tf":
            x = self.transformer_encoder(x)
        elif self.mode == "rnn":
            x, _ = self.rnn(x)
        else:
            assert self.mode == "neither", "Ensure that mode is either 'rnn', 'tf' or 'neither'."
        return x  # Shape: (B, seq_output_length, num_filters*2)


class StrokeEncoder(nn.Module):
    def __init__(self, input_dim=2, d_model=128, mode = "tf", nhead = 4, num_layers = 3):
        super().__init__()
        assert mode in ("rnn", "tf", "neither")
        self.mode = mode
        self.single_encoder = SingleStrokeEncoder(input_dim=input_dim,d_model=d_model)
        self.multi_encoder = MultiStrokeEncoder(input_dim=d_model, num_filters=64,
                                                nhead=nhead, num_layers=num_layers, mode = mode,
                                                kernel_size=3, stride=2, pool_stride=2)
        
    def forward(self, stroke, mask=None):
        # Condition only on the previous stroke  
        if len(stroke.shape) == 3:
            stroke = stroke.unsqueeze(1)          
        B, N, L, d = stroke.size()
    
        embeds = self.single_encoder(stroke, mask) # B x N x L x d_model with N = 1 or 24
        embeds = embeds.view(B,N*L,-1) # B x NL x d_model
        if N > 1:
            embeds = self.multi_encoder(embeds) # B x M x d_model
        
        return embeds
