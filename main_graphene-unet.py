from model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from time import time
from dataset import get_loaders
from utils import *

start = time()

print('UNet model for graphene dataset')

DEVICE = 'cuda'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300


graphene_ds = np.load('./datasets/graphene_dataset.npz')

images, labels = graphene_ds['images'], graphene_ds['labels']


def train_epoch(device, loader, model, optimizer, loss_batch, scaler):

    running_loss = 0.
    last_loss = 0.
    
    

    loop = tqdm(loader)
    for batch_idx, (features, targets) in enumerate(loop):

        features = features.to(device)
        targets = targets.to(device)

        #forward

        with torch.cuda.amp.autocast():
            predictions = model(features)
            loss = loss_batch(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())
        running_loss += loss.item() * features.size(0)
        # print(f'Minibatch loss:{minibatch_loss}')
    last_loss = running_loss / len(loader.dataset)
    model.loss_acc["train_loss"].append(np.array(last_loss))
    return last_loss


def train():

    best_vloss = float('inf')
    counter = 0
    patience = 70

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader= get_loaders(images=images, 
                                        labels=labels,
                                        split=True,
                                        num_classes=1,
                                        batch_size=16,
                                        num_workers=4,
                                        pin_memory=True)

    for epoch in range(NUM_EPOCHS):


        print(f'Epoch: {epoch+1} with early stopping counter in {counter} \n')
        model.train()
        avg_loss= train_epoch(DEVICE, train_loader, model, optimizer, loss_fn, scaler)
        #check acc - training set
        train_acc = compute_binary_iou(train_loader, model, device=DEVICE)
        model.loss_acc['train_acc'].append(np.array(train_acc))
        
        running_vloss = 0.0

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(val_loader):
                vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item() * vinputs.size(0) 

        avg_vloss = running_vloss / len(val_loader.dataset)
        model.loss_acc["val_loss"].append(np.array(avg_vloss))

        #check acc - validation set
        val_acc = compute_binary_iou(val_loader, model, device=DEVICE)
        model.loss_acc['val_acc'].append(np.array(val_acc))

        print('Training loss: {} \n Validation loss: {} \n'.format(avg_loss, avg_vloss))
        print('Training acc: {} \n Validation acc: {} \n'.format(train_acc, val_acc))
        
        save_loss(model=model, name='./graphene_unet_results/graphene_unet_loss_acc')
    
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            counter = 0
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, name='./graphene_unet_results/graphene_unet_best_model_checkpoint.pth.tar')
        else:
            counter += 1
        
        if counter > patience:
            print('Model stopped at {} epochs.'.format(epoch))
            break
    if epoch == NUM_EPOCHS-1:
        print('Model finished all epochs!')


if __name__ == "__main__":
    train()
    print('Total Training Time: %.2f min' % ((time() - start)/60))
