import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VariationalAutoEncoder
from torchvision.utils import save_image

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR_RATE = 3e-4 

# Dataset Loading
dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Initialization
model = VariationalAutoEncoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction='sum')

# Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(dataloader))
    for batch_idx, (x, _) in loop:
        # Forward Pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)
        
        # Loss Computation
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) 
        loss = reconstruction_loss + kl_div

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        loop.set_description(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        loop.set_postfix(loss=loss.item())

# Save the model
torch.save(model.state_dict(), 'vae.pth')

# Load the model
model.load_state_dict(torch.load('vae.pth'))

def inference(digit, num_examples):
    images = []
    idx = 0
    for x, y in dataset:
        if y == digit:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encoding_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIM).to(DEVICE))
        encoding_digit.append((mu, sigma))

    mu, sigma = encoding_digit[digit]

    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        out = model.decode(z_reparametrized)
        save_image(out.view(1, 1, 28, 28), f'images/digit_{digit}_example_{example}.png')

for digit in range(10):
    inference(digit, 5)



