import torch
import torch.nn as nn

from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer, device):

    nb_batchs = dataloader.batch_size
    model.train()
    for idx, (x, label) in enumerate(tqdm(dataloader)):

        pred = model(x.to(device))
        loss = loss_fn(pred, label.long().to(device))
        
        #Back Pass
        loss.backward()
        
        _, predicted = torch.max(pred, 1)
        
        #Weights Update
        optimizer.step()
        optimizer.zero_grad()
        
        acc = ((predicted == label.to(device)).sum())/nb_batchs

        writer.add_scalar('training_loss', loss.item(), (epoch*len(dataloader)) + idx)
        print(f'training-loss : {loss.item():>7f} | [{(idx+1)}/ {len(dataloader)}]')
        writer.add_scalar('Accuracy_train', acc, (epoch*len(dataloader)) + idx)
        print(f'Accuracy : {acc:>7f} | [{(idx+1)}/ {len(dataloader)}]')



def test_loop(dataloader, model, loss_fn, epoch, writer, device):
    test_loss = 0
    correct = 0
    nb_batchs = dataloader.batch_size
    model.eval()
    with torch.no_grad():
        for idx, (x, label) in enumerate(tqdm(dataloader)):
            pred = model(x.to(device))
            test_loss += loss_fn(pred, label.long().to(device)).item()
            
            _, predicted = torch.max(pred, 1)
            
            correct += (predicted==label.to(device)).sum()/nb_batchs
            
        test_loss /= len(dataloader)
        writer.add_scalar('testing_loss', test_loss, epoch)
        writer.add_scalar('Accuracy_test', correct/len(dataloader), epoch)
        print(f'Test MSE loss : {test_loss:>7f}, Accuracy : {correct/len(dataloader):>7f}')
