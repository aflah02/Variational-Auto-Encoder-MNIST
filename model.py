import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # Decoder
        self.z_hid = nn.Linear(z_dim, h_dim)
        self.hid_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_hid(z))
        x = self.hid_img(h)
        return torch.sigmoid(x) 
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == '__main__':
    x = torch.randn(4, 784)
    model = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = model(x)
    print(x_reconstructed.shape, mu.shape, sigma.shape)
