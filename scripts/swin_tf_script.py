import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn

from model.model_V2 import SwinTransformer
from data.dataset import CustomDataset

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
from train.train import train_loop, test_loop

from torch.utils.data import DataLoader, WeightedRandomSampler



if __name__ == "__main__":
    file_path = '/home/sajjad/Desktop/Face_Classification/data'
    labels = 'label_train.txt'


    nb_images = 111430
    width = 56  # Replace with the actual image width
    height = 56  # Replace with the actual image height
    channels = 3

    with open(f"{file_path}/db_train.raw", 'rb') as file:
        train_data = file.read()
    labels = np.loadtxt(f"{file_path}/label_train.txt", dtype=np.int8)


    # Assuming the data represents a grayscale image
    data = np.frombuffer(train_data, dtype=np.uint8).reshape(nb_images, height, width, channels).astype(np.float32)
    

    class_sample_count = np.unique(labels, return_counts=True)[1]

    train_data, val_data, train_lab, val_lab = train_test_split(data, labels, stratify=labels, test_size=10130/111430) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32   
    
    parameters = {
        'kernel_size' : (2, 2),
        'hidden_emb' : 48, 
        'nb_classes' : 2, 
        'nb_channels' : 3, 
        'data_size' : (56, 56), 
        'norm_layer' : nn.LayerNorm,
        'loss' : nn.BCELoss(),

        'num_layers' : 4,
        'num_heads' : 3,
        'head_dim' : 16, 
        'window_size' : (2, 2),
        'relative_pos_embedding'  : True,
        'attn_drop_rate' : 0.1,
        'proj_drop_rate' : 0.1, 
        'mlp_dim' : 48 * 2,
        'batch_size' : 64,
        'device' : device,
        'lr': 1e-5,
        'weight_decay' : 1e-6,
        'nb_epochs': 10,
        'enc_layers': 3
    }
    


    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[train_lab])
    sampler = WeightedRandomSampler(samples_weight, 2*len(samples_weight), replacement=True)

    train_dataset = CustomDataset(data = train_data, labels = train_lab)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=sampler)


    val_dataset = CustomDataset(data = val_data, labels = val_lab)
    val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=True)
    
    
    writer = SummaryWriter(f"runs/modele4 == batch_size = {parameters['batch_size']}, kernel_size = {parameters['kernel_size']}, \
        lr = {parameters['lr']}, window_size = {parameters['window_size']}, epochs = {parameters['nb_epochs']}")
    
    
    #save model parameters
    with open(f"model_N4_params.pkl", "wb") as file:
        pickle.dump(parameters, file)

    
    model = SwinTransformer(parameters).to(dtype = dtype, device= device)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    nb_epochs = parameters['nb_epochs']
    
    
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer, epoch, writer, device)
        test_loop(val_dataloader, model, criterion, epoch, writer, device)
    
    #save the model weights
    torch.save(model.state_dict(), f"model_N4.pt")
    
    print('Done')
    
    