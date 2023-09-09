import os
import numpy as np 
import torch
import torch.nn as nn

from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer):

    nb_batchs = dataloader.batch_size
    model.train()
    for idx, (x, label) in enumerate(tqdm(dataloader)):

        pred = model(x)
        loss = loss_fn(pred, label.long())
        
        #Back Pass
        loss.backward()
        
        _, predicted = torch.max(pred, 1)
        
        #Weights Update
        optimizer.step()
        optimizer.zero_grad()
        
        
        writer.add_scalar('training_loss', loss.item(), (epoch*len(dataloader)) + idx)
        print(f'training-loss : {loss.item():>7f} | [{(idx+1)}/ {len(dataloader)}]')
        print(f'Accuracy : {((predicted == label).sum())/nb_batchs:>7f} | [{(idx+1)}/ {len(dataloader)}]')



def test_loop(dataloader, model, loss_fn, epoch, writer):
    test_loss = 0
    correct = 0
    nb_batchs = dataloader.batch_size
    model.eval()
    with torch.no_grad():
        for idx, (x, label) in enumerate(tqdm(dataloader)):
            pred = model(x)
            test_loss += loss_fn(pred, label.long()).item()
            
            _, predicted = torch.max(pred, 1)
            
            correct += (predicted==label).sum()/nb_batchs
            
        test_loss /= len(dataloader)
        writer.add_scalar('testing_loss', test_loss, epoch)
        print(f'Test MSE loss : {test_loss:>7f}, Accuracy : {correct:>7f}')
