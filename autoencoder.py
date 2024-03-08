import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout, dropout_layers):
        super(Autoencoder, self).__init__()

        ### Encoder
            # Primera capa
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU6(),
            nn.Dropout(dropout)
        )
        
            # Capas intermedias
        for i in range(len(hidden_dims) - 1):
                #Solo aplicamos el dropout a las primeras n capas
            if i < dropout_layers-2:
                layer = nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.LeakyReLU(), nn.Dropout(dropout))
            else:
                layer = nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.LeakyReLU())
            self.encoder.add_module(f"layer_{i}", layer) 
        
            #Última capa
        self.encoder.add_module(f"Last", nn.Linear(hidden_dims[i+1], latent_dim))
        

        ### Decoder
            #Primera capa
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[len(hidden_dims)-1]),
            nn.LeakyReLU())
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            # En este caso el dropout solo a las ultimas n capas 
            if i <= dropout_layers - 1:
                layer = nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i-1]), nn.LeakyReLU(), nn.Dropout(dropout))
            else:
                layer = nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i-1]), nn.LeakyReLU())
            self.decoder.add_module(f"layer_{i}", layer) 
        
            #Última capa
        self.decoder.add_module(f"Last", nn.Sequential(nn.Linear(hidden_dims[0], input_dim), nn.Sigmoid()))

    def forward(self, x):
        #print(x.shape)
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

# Compute model paramters
# def compute_model_params(model):
#   params = 0
#   for p in model.parameters():
#     params+= p.numel()
#   return params

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datos
data = pd.read_csv('/home/gribes/Project/data/processedWaveformsDFs/Umbilical Artery IMPACT.csv')
data = abs(data.drop(columns=["ID"],axis=1)).astype(float)



# input_train= data.head(-2).values
# input_test= data.tail(2).values

# Split the data into training and testing sets
input_train, input_test = train_test_split(data, test_size=0.1)#, random_state=42)



# Scale the input data
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)



# Convert the data to PyTorch tensors
input_train = torch.tensor(input_train, dtype=torch.float32).to(device)
input_test = torch.tensor(input_test, dtype=torch.float32).to(device)

# Initialize the autoencoder
#(self, input_dim, hidden_dims, latent_dim, dropout, dropout_layers)
input_dim = input_train.shape[1]

#hidden_dimensions = [146,129,118,105,80,64,52,47,33,20,11,8]
#hidden_dimensions = [140,120,100,80,60,40,20,10,8]
hidden_dimensions = [150, 140, 130, 120, 110, 100, 90, 80, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]
latent_dim = 4
dropout = 0.2
dropout_layers=15


autoencoder = Autoencoder(input_dim, hidden_dimensions, latent_dim, dropout=dropout, dropout_layers=dropout_layers).to(device)

    # Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0003)#3e-4)

    # Training loop
num_epochs = 1000
reconstruction_loss_ev = []
evaluation_loss_ev = []


for epoch in range(num_epochs):
        # Forward pass
    reconstructed_input = autoencoder(input_train)
    reconstructed_test = autoencoder(input_test)

        # Calculate loss
        #loss = criterion(reconstructed_input, input_train)
    loss = criterion(reconstructed_input, input_train)
    evaluation_loss = criterion(reconstructed_test, input_test)
        
    reconstruction_loss_ev.append(loss.cpu().detach().numpy())
    evaluation_loss_ev.append(evaluation_loss.cpu().detach().numpy())

        # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

        # Print loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    


plt.plot(reconstruction_loss_ev)
plt.plot(evaluation_loss_ev)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Losses evolution')
plt.grid(True)
plt.show()
plt.savefig('out/training_losses_ev.png')

# plt.figure(figsize=(8, 6))
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title('Evaluation loss evolution')
# plt.grid(True)
# plt.show()
# plt.savefig('evaluation_loss_ev.png')

#print(input_test[0].shape)

# Test
with torch.no_grad():

        # Convert input_train tensor to numpy array
    input_train_np = input_train.cpu().numpy()

    # Train PCA
    pca = PCA(n_components=4)  # Reduce to 3 features
    pca.fit(input_train_np)

    # Initialize an empty list to store losses for each sample
    losses_autoencoder = []
    losses_pca = []

    # Define MSE loss function
    mse_loss = torch.nn.MSELoss()

    # Iterate over each sample in the input_test tensor
    for sample_idx in range(input_test.size(0)):
        # Get the sample from input tensor
        single_input_test = input_test[sample_idx].unsqueeze(0)  # shape: (1, features)
        
        # Pass the sample through the autoencoder
        reconstructed_sample_autoencoder = autoencoder(single_input_test)
        
        # Compute loss for the sample using autoencoder
        loss_autoencoder = mse_loss(reconstructed_sample_autoencoder, single_input_test)
        
        # Print or store the loss for autoencoder
        print(f"Autoencoder Loss for sample {sample_idx + 1}: {loss_autoencoder.item()}")
        losses_autoencoder.append(loss_autoencoder.item())
        
        # Plot the original vs reconstructed data for autoencoder
        denormalized_input = scaler.inverse_transform(single_input_test.cpu().numpy())
        denormalized_reconstructed_autoencoder = scaler.inverse_transform(reconstructed_sample_autoencoder.cpu().numpy())

        # Perform PCA reconstruction for the current sample
        reconstructed_sample_pca = pca.inverse_transform(pca.transform(single_input_test.cpu().numpy()))
        
        # Convert reconstructed_sample_pca to torch tensor
        reconstructed_sample_pca = torch.tensor(reconstructed_sample_pca, dtype=torch.float32)
        
        denormalized_reconstructed_sample_pca = scaler.inverse_transform(reconstructed_sample_pca)
        # Compute reconstruction error for PCA using MSELoss
        loss_pca = mse_loss(single_input_test, reconstructed_sample_pca)
        
        # Print or store the loss for PCA
        print(f"PCA Reconstruction Error for sample {sample_idx + 1}: {loss_pca.item()}")
        losses_pca.append(loss_pca.item())
        
        # Plot the original vs PCA-reconstructed data for the current sample
        plt.figure(figsize=(8, 6))
        plt.plot(denormalized_input.squeeze(), label="Input")
        plt.plot(denormalized_reconstructed_autoencoder.squeeze(), label="Autoencoder Output")
        plt.plot(denormalized_reconstructed_sample_pca.squeeze(), label="PCA Output")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f'Original vs PCA-Reconstructed Data for sample {sample_idx + 1}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'out/compare_plot_pca_sample_{sample_idx + 1}.png')
        plt.show()

    # After processing all samples, you can compute the average loss or any other aggregate metric if needed
    average_loss_autoencoder = torch.mean(torch.tensor(losses_autoencoder))
    average_loss_pca = torch.mean(torch.tensor(losses_pca))
    print(f"Average Autoencoder Loss: {average_loss_autoencoder}")
    print(f"Average PCA Reconstruction Error: {average_loss_pca}")









    # Initialize an empty list to store losses for each sample
    # Convert input_train tensor to numpy array
    # input_train_np = input_train.cpu().numpy()

    # # Train PCA
    # pca = PCA(n_components=3)  # Reduce to 3 features
    # pca.fit(input_train_np)

    # # Initialize an empty list to store losses for each sample
    # losses_autoencoder = []
    # losses_pca = []

    # # Iterate over each sample in the input_test tensor
    # for sample_idx in range(input_test.size(0)):
    #     # Get the sample from input tensor
    #     single_input_test = input_test[sample_idx].unsqueeze(0)  # shape: (1, features)
        
    #     # Pass the sample through the autoencoder
    #     reconstructed_sample_autoencoder = autoencoder(single_input_test)
        
    #     # Compute loss for the sample using autoencoder
    #     loss_autoencoder = criterion(reconstructed_sample_autoencoder, single_input_test)
        
    #     # Print or store the loss for autoencoder
    #     print(f"Autoencoder Loss for sample {sample_idx + 1}: {loss_autoencoder.item()}")
    #     losses_autoencoder.append(loss_autoencoder.item())
        
    #     # Plot the original vs reconstructed data for autoencoder
    #     denormalized_input = scaler.inverse_transform(single_input_test.cpu().numpy())
    #     denormalized_reconstructed_autoencoder = scaler.inverse_transform(reconstructed_sample_autoencoder.cpu().numpy())
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(denormalized_input.squeeze(), label="Input")
    #     plt.plot(denormalized_reconstructed_autoencoder.squeeze(), label="Autoencoder Output")
    #     plt.xlabel("Time")
    #     plt.ylabel("Value")
    #     plt.title(f'Original vs Autoencoder-Reconstructed Data for sample {sample_idx + 1}')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f'out/compare_plot_autoencoder_sample_{sample_idx + 1}.png')
    #     plt.show()

    #     # Perform PCA reconstruction for the current sample
    #     reconstructed_sample_pca = pca.inverse_transform(pca.transform(single_input_test.cpu().numpy()))
        
    #     # Compute reconstruction error for PCA
    #     loss_pca = torch.nn.MSELoss(single_input_test, reconstructed_sample_pca)
        
    #     # Print or store the loss for PCA
    #     print(f"PCA Reconstruction Error for sample {sample_idx + 1}: {loss_pca}")
    #     losses_pca.append(loss_pca)
        
    #     # Plot the original vs PCA-reconstructed data for the current sample
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(denormalized_input.squeeze(), label="Input")
    #     plt.plot(reconstructed_sample_pca.squeeze(), label="PCA Output")
    #     plt.xlabel("Time")
    #     plt.ylabel("Value")
    #     plt.title(f'Original vs PCA-Reconstructed Data for sample {sample_idx + 1}')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f'out/compare_plot_pca_sample_{sample_idx + 1}.png')
    #     plt.show()

    # # After processing all samples, you can compute the average loss or any other aggregate metric if needed
    # average_loss_autoencoder = torch.mean(torch.tensor(losses_autoencoder))
    # average_loss_pca = torch.mean(torch.tensor(losses_pca))
    # print(f"Average Autoencoder Loss: {average_loss_autoencoder}")
    # print(f"Average PCA Reconstruction Error: {average_loss_pca}")

#count_parameters(autoencoder)
#print("Parametres:",compute_model_params(autoencoder))