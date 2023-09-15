import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from model.model_V2 import SwinTransformer
from data.dataset import CustomDataset
from train.train import train_loop, test_loop

from einops import rearrange


if __name__ == "__main__":
    file_path = '/home/sajjad/Desktop/Face_Classification/data'
    labels = 'label_train.txt'


    nb_images = 111430
    nb_test_images = 10130
    width = 56  
    height = 56  
    channels = 3
    data_augment = False

    # Load the data
    with open(f"{file_path}/db_train.raw", 'rb') as file:
        data = file.read()
    labels = np.loadtxt(f"{file_path}/label_train.txt", dtype=np.int8)

    data = np.frombuffer(data, dtype=np.uint8).reshape(nb_images, height, width, channels).astype(np.float32)
    
    # Count the number of images in each class
    class_sample_count = np.unique(labels, return_counts=True)[1]
    # The data in imbalanced

    # Split the data to the same proportion as train and test data
    train_data, val_data, train_lab, val_lab = train_test_split(
        data, labels, stratify=labels, test_size=nb_test_images/nb_images) 
    train_data = torch.from_numpy(train_data)
    train_data = rearrange(train_data, 'b h w c -> b c h w')


    # the Parameters we need to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32   
    parameters = {
        'kernel_size' : (2, 2),
        'hidden_emb' : 48, 
        'nb_classes' : 2, 
        'nb_channels' : 3, 
        'data_size' : (56, 56), 
        'norm_layer' : nn.LayerNorm,
        'loss' : nn.CrossEntropyLoss(),

        'num_layers' : 4,
        'num_heads' : 3,
        'head_dim' : 16, 
        'window_size' : (2, 2),
        'relative_pos_embedding'  : True,
        'attn_drop_rate' : 0.2,
        'proj_drop_rate' : 0.2, 
        'mlp_dim' : 48 * 2,
        'batch_size' : 128,
        'device' : device,
        'lr': 1e-4,
        'weight_decay' : 1e-6,
        'nb_epochs': 10,
        'enc_layers': 2
    }
    

    # To overcome with imbalanced data
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[train_lab])
    sampler = WeightedRandomSampler(samples_weight, 2*len(samples_weight), replacement=True)
    
    # Transforms done on images if we want to augment the datset
    transform = T.Compose([
            lambda x : x.float(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            lambda x : x.permute(1,2,0)
    ])

    transform_augment = T.Compose([
            T.TrivialAugmentWide(5),
            lambda x : x.float(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
            lambda x : x.permute(1,2,0)
    ])


    # Create the dataloaders
    train_dataset = CustomDataset(data = train_data, labels = train_lab, transforms=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=sampler)

    val_dataset = CustomDataset(data = val_data, labels = val_lab)
    val_dataloader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=True)
    
    # Essential for TensorBoard Tool 
    writer = SummaryWriter(f"runs/modele7 == batch_size = {parameters['batch_size']}, kernel_size = {parameters['kernel_size']}, \
        lr = {parameters['lr']}, window_size = {parameters['window_size']}, epochs = {parameters['nb_epochs']}, data_augment = {data_augment}")
    
    
    #save model parameters
    with open(f"model_N7_params.pkl", "wb") as file:
        pickle.dump(parameters, file)

    # The Swin Transformer Model
    model = SwinTransformer(parameters).to(dtype = dtype, device= device)

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    nb_epochs = parameters['nb_epochs']
    # The optimization loops
    for epoch in range(nb_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataloader, model, criterion, optimizer, epoch, writer, device)
        test_loop(val_dataloader, model, criterion, epoch, writer, device)
    

    if data_augment :
        train_dataset_augment = CustomDataset(data = train_data, labels = train_lab, transforms=transform_augment)
        train_dataloader_augment = DataLoader(train_dataset, batch_size=parameters['batch_size'], sampler=sampler)


        for epoch in range(nb_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loop(train_dataloader_augment, model, criterion, optimizer, epoch + nb_epochs, writer, device)
            test_loop(val_dataloader, model, criterion, epoch + nb_epochs, writer, device)

    #save the model weights
    torch.save(model.state_dict(), f"model_N7.pt")
    
    print('Done')
    