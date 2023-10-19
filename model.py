# Importing necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
# If CUDA is available, use CUDA. Else, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class to apply positional encoding to input tensor
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Dropout layer with given dropout rate
        self.dropout = nn.Dropout(p=dropout)
        # Generate a tensor filled with positional information. Unsqueeze(1) adds an extra dimension.
        position = torch.arange(max_len).unsqueeze(1).float()
        # Generate the div_term used in the positional encoding calculation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        # Prepare a zero tensor to hold positional encoding values
        pe = torch.zeros(max_len, 1, d_model)
        # Apply sine to even indices and cosine to odd indices in the tensor
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register pe as a buffer that should not to be considered a model parameter. 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add positional encoding to input tensor and apply dropout
       x = x + self.pe[:x.size(0), :]
       return self.dropout(x)

# Class for a Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout, output_dim):
        super(TransformerModel, self).__init__()

        # Embedding layer that maps the input values to embeddings of 'input_dim' dimensions
        self.embed = nn.Embedding(input_dim, input_dim)
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        # Define the transformer encoder layer with specified parameters
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=4*input_dim, dropout=dropout)
        # Define the transformer encoder with the specified number of layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Linear layer to map the output of the transformer to output_dim
        self.decoder = nn.Linear(input_dim, output_dim)
       
    def forward(self, src):
        # Pass the input through embedding layer and scale it
        src = self.embed(src) * torch.sqrt(torch.tensor(self.embed.embedding_dim))
        # Apply positional encoding to the embedded input
        src = self.pos_encoder(src)
        # Pass the positional encoded input through the transformer encoder
        output = self.transformer_encoder(src)
        # Apply linear transformation to the output of transformer encoder
        output = self.decoder(output)
        # Apply softmax to output values to get probabilities
        return F.log_softmax(output, dim=-1)
