import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, hidden_dim7, hidden_dim8, hidden_dim9, latent_dim, dropout):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim4, hidden_dim5),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim5, hidden_dim6),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim6, hidden_dim7),
            nn.ReLU6(),
            nn.Linear(hidden_dim7, hidden_dim8),
            nn.ReLU6(),
            nn.Linear(hidden_dim8, hidden_dim9),
            nn.ReLU6(),
            nn.Linear(hidden_dim9, latent_dim)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim9),
            nn.ReLU6(),
            nn.Linear(hidden_dim9, hidden_dim8),
            nn.ReLU6(),
            nn.Linear(hidden_dim8, hidden_dim7),
            nn.ReLU6(),
            nn.Linear(hidden_dim7, hidden_dim6),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim6, hidden_dim5),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim5, hidden_dim4),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim4, hidden_dim3),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()  # Use sigmoid activation for reconstruction
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

# Compute model paramters
def compute_model_params(model):
  params = 0
  for p in model.parameters():
    params+= p.numel()
  return params


# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data from a CSV file
data = pd.read_csv('/home/gribes/Project/data/processedWaveformsDFs/Umbilical Artery IMPACT.csv')
data = abs(data.drop(columns=["ID"],axis=1)).astype(float)
input_train= data.head(-2).values
input_test= data.tail(2).values

# Split the data into training and testing sets
#input_train, input_test = train_test_split(input_data, test_size=0.2, random_state=42)

# Scale the input data
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)

# Convert the data to PyTorch tensors
input_train = torch.tensor(input_train, dtype=torch.float32).to(device)
input_test = torch.tensor(input_test, dtype=torch.float32).to(device)

# Initialize the autoencoder
input_dim = input_train.shape[1]


hidden_dim1 = 140
hidden_dim2 = 120
hidden_dim3 = 100
hidden_dim4 = 80
hidden_dim5 = 60
hidden_dim6 = 40
hidden_dim7 = 20
hidden_dim8 = 10
hidden_dim9 = 5
latent_dim = 3
dropout_rate = 0.25

autoencoder = Autoencoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, hidden_dim7, hidden_dim8, hidden_dim9, latent_dim, dropout_rate).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()#r2_score
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=3e-4)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    reconstructed_input = autoencoder(input_train)

    # Calculate loss
    loss = criterion(reconstructed_input, input_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#print(f"Input test: {scaler.inverse_transform(input_test.cpu().numpy())}")

# Test
with torch.no_grad():
    reconstructed_test = autoencoder(input_test)
    denormalized_reconstructed_test = scaler.inverse_transform(reconstructed_test.cpu().numpy())
    #print(f"Denormalized reconstructed test: {denormalized_reconstructed_test}")
    test_loss = criterion(reconstructed_test, input_test)
    
    #test_reconstruction_error = r2_score(input_test, denormalized_reconstructed_test)#, multioutput='variance_weighted')
    print(f'Test Loss: {test_loss.item():.4f}')

print("Parametres:",compute_model_params(autoencoder))