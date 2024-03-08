from autoencoder import Autoencoder
from dataset_p import dataset_p
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim




if __name__ == "__main__":
    #Dims i parametres (provisionals)
    input_dim = 160
    hidden_dim = 100
    latent_dim = 60
    
    num_epochs = 2000
    batch_size = 32
    learning_rate = 0.00005

    # Cast the input tensor to the same data type as the layers in the Autoencoder module
    torch.set_default_tensor_type(torch.DoubleTensor)

    #Init autoencoder
    autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim)

    # Train data
    umbilical_artery_data = pd.read_csv("/home/gribes/Project/data/processedWaveformsDFs/Umbilical Artery IMPACT.csv")
    umbilical_artery_data = abs(umbilical_artery_data.drop(columns=["ID"],axis=1)).astype(float)
    train_data = dataset_p(umbilical_artery_data)
    
    
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define the optimizer and loss function
    optimizer_f = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    loss_f = nn.BCELoss(reduction="sum")#F.mse_loss
    
    
    # Train the autoencoder
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            decoded = autoencoder(batch)

            # Compute the loss
            loss = loss_f(decoded, batch)

            # Backward pass
            optimizer_f.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer_f.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        



    input_data = torch.from_numpy(umbilical_artery_data.iloc[0].values).double()
    print("Input data:", input_data)

    # Encoding
    encoded_data = autoencoder.encode(input_data)
    print("Encoded data:", encoded_data)

    # Decoding
    decoded_data = autoencoder.decode(encoded_data)
    print("Decoded data:", decoded_data)