import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from dataset_p import dataset_p

# Load your time series data




# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * 2),  # mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 160),
            nn.Sigmoid(),  # ensure output values are in the range of [0, 1]
        )

    def encode(self, x):
        mu, log_var = self.encoder(x).split(32, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

vae = VAE()

# Define the loss function
def loss_function(reconstructed_x, x, mu, log_var):
    reconstruction_loss = nn.MSELoss()(reconstructed_x, x)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kld_loss /= x.shape[0] * 160  # normalize the KL-divergence term
    return reconstruction_loss + kld_loss




if __name__ == "__main__":

    # Train data
    umbilical_artery_data = pd.read_csv("/home/gribes/Project/data/processedWaveformsDFs/Umbilical Artery IMPACT.csv")
    umbilical_artery_data = abs(umbilical_artery_data.drop(columns=["ID"],axis=1)).astype(float)
    train_data = dataset_p(umbilical_artery_data)
    
    
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)


# Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)


# Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    loss_f = nn.BCELoss(reduction="sum")#F.mse_loss

    # Train the VAE
    num_epochs = 1000
    for epoch in range(num_epochs):
        for data_batch in dataloader:
            data_batch = data_batch.to(device)

            optimizer.zero_grad()

            reconstructed_x, mu, log_var = vae(data_batch)
            loss = loss_function(reconstructed_x, data_batch, mu, log_var)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the VAE
    with torch.no_grad():
        reconstructed_x, mu, log_var = vae(train_data.to(device))
        mse = nn.MSELoss()(reconstructed_x, train_data.to(device))
        print('Mean Squared Error:', mse.item())