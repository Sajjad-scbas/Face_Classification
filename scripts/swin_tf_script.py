import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from model.model import SwinTransformer
from data.dataset import CustomDataset

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from train.train import train_loop, test_loop

from torch.utils.data import DataLoader



if __name__ == "__main__":
    file_path = '/Users/scbas/Downloads/IdemiaExerciceML/data'
    labels = 'label_train.txt'


    nb_images = 111430
    width = 56  # Replace with the actual image width
    height = 56  # Replace with the actual image height
    channels = 3

    with open(f"{file_path}/db_train.raw", 'rb') as file:
        train_data = file.read()
    labels = np.loadtxt(f"{file_path}/label_train.txt", dtype = np.float32)


    # Assuming the data represents a grayscale image
    data = np.frombuffer(train_data, dtype=np.uint8).reshape(nb_images, height, width, channels).astype(np.float32)
    
    train_data, val_data, train_lab, val_lab = train_test_split(data, labels, test_size=10130/111430) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32   
    
    parameters = {
        'kernel_size' : (4, 4),
        'hidden_emb' : 48, 
        'nb_classes' : 2, 
        'nb_channels' : 3, 
        'data_size' : (56, 56), 
        'norm_layer' : nn.LayerNorm,
        'loss' : nn.BCELoss(),

        'num_layers' : 2,
        'num_heads' : 3,
        'head_dim' : 16, 
        'window_size' : (4, 4),
        'relative_pos_embedding'  : True,
        'attn_drop_rate' : 0.2,
        'proj_drop_rate' : 0.2, 
        'mlp_dim' : 48 * 2,
        'batch_size' : 128,
        'device' : device,
        'lr': 5e-6,
        'weight_decay' : 3e-7,
        'nb_epochs': 10,
    }
    
    train_dataset = CustomDataset(data = train_data, labels = train_lab)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)

    val_dataset = CustomDataset(data = val_data, labels = val_lab)
    val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=True)
    
    
    writer = SummaryWriter(f"runs/modele0 == batch_size = {parameters['batch_size']}, kernel_size = {parameters['kernel_size']}, \
        lr = {parameters['lr']}, window_size = {parameters['window_size']}, epochs = {parameters['nb_epochs']}")
    
    

    
    
    model = SwinTransformer(parameters).to(dtype = dtype, device= device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
    criterion = nn.CrossEntropyLoss()
    
    nb_epochs = parameters['nb_epochs']
    
    
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer, epoch, writer)
        test_loop(val_dataloader, model, criterion, epoch, writer)
    
        #save the model weights
        torch.save(model.state_dict(), f"model0.pt")
    
    print('Done')
    
    