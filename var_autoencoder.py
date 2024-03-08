import torch
from torch import nn
import pandas as pd
from dataset_p import dataset_p
import torch.optim as optim

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 100, z_dim = 60): #en el meu cas la input dimension sera 160
        super().__init__()

        #Encoder
        self.view_to_hid = nn.Linear(input_dim, h_dim)
        self.hid_to_mu = nn.Linear(h_dim, z_dim)
        self.hid_to_sigma = nn.Linear(h_dim, z_dim)
        
        #Decoder
        self.z_to_hid = nn.Linear(z_dim, h_dim)
        self.hid_to_view = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU(inplace = True)
        
        
    def encode(self,x): #q_phi(z|x)
        h = self.relu(self.view_to_hid(x))
        
        mu, sigma = self.hid_to_mu(h),self.hid_to_sigma(h)
        
        return mu, sigma
    
    def decode(self, z): #p_theta(x|z)
        h = self.relu(self.z_to_hid(z))
        return torch.sigmoid(self.hid_to_view(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    
    
    
# if __name__ == "__main__":
    
#     #Dims i parametres (provisionals)
#     input_dim = 160
#     hidden_dim = 100
#     latent_dim = 80
    
#     num_epochs = 2000
#     batch_size = 32
#     learning_rate = 0.0001
    
#     # Cast the input tensor to the same data type as the layers in the Autoencoder module
#     torch.set_default_tensor_type(torch.DoubleTensor)
    
#     umbilical_artery_data = pd.read_csv("/home/gribes/Project/data/processedWaveformsDFs/Umbilical Artery IMPACT.csv")
#     umbilical_artery_data = abs(umbilical_artery_data.drop(columns=["ID"],axis=1)).astype(float)
#     train_data = dataset_p(umbilical_artery_data)
    
#     dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

#     VAE = VariationalAutoEncoder(input_dim)
    
#     # Define the optimizer and loss function
#     optimizer_f = optim.Adam(VAE.parameters(), lr=learning_rate)
#     loss_f = nn.BCELoss(reduction="sum")#F.mse_loss
    
#     #Training
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             x_reconstructed, mu, sigma = VAE(batch)
            
#             #Loss
#             reconstruction_loss = loss_f(x_reconstructed, batch)
#             kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            
#             #Backprop
#             loss = reconstruction_loss + kl_div
#             optimizer_f.zero_grad()
#             loss.backward()
#             optimizer_f.step()
            
    
    
#     input_data = torch.from_numpy(umbilical_artery_data.iloc[0].values).double()
#     print("Input data:", input_data)

#     # Encoding
#     encoded_data = VAE.encode(input_data)
#     print("Encoded data:", encoded_data)

#     # Decoding
#     decoded_data = VAE.decode(encoded_data)
#     print("Decoded data:", decoded_data)
if __name__ == "__main__":
        
    x = torch.randn(4, 160)
    # print(x)
    vae = VariationalAutoEncoder(input_dim=160)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
    