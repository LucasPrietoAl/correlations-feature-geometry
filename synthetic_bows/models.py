import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, use_relu=False, tie_weights=True, use_bias=False):
        """
        If tie_weights is True, then the decoder will use the transpose of the encoder weight
        along with a free bias parameter. Otherwise, the decoder has independent weights.
        """
        super().__init__()
        self.use_relu = use_relu
        self.tie_weights = tie_weights
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        
        if self.tie_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=use_bias)
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x):
        h = self.encoder(x)
        if self.tie_weights:
            x_hat = F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        else:
            x_hat = self.decoder(h)
        if self.use_relu:
            x_hat = F.relu(x_hat)
        return x_hat, h
    
    def get_latent(self, x):
        return self.encoder(x)
    
    def get_representation_matrix(self, off_diagonal=False):
        if self.tie_weights:
            representation = self.encoder.weight.t() @ self.encoder.weight
        else:
            representation = self.decoder.weight @ self.encoder.weight
        if off_diagonal:
            representation[torch.eye(len(representation)).bool()] = 0
        return representation
