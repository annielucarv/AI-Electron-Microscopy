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

print('UNet model for MoS2 dataset')

DEVICE = 'cuda'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500


mos2_ds = np.load('./datasets/mos2_dataset.npz')

images, labels = mos2_ds['images'], mos2_ds['labels']

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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
            # minibatch_loss.append(loss.detach().cpu().numpy())

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
    epoch_fin = 0

    model = UNet(in_channels=1, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(images=images, 
                                        labels=labels, 
                                        split=True,
                                        num_classes=3,
                                        batch_size=16,
                                        num_workers=4,
                                        pin_memory=True)

    for epoch in range(NUM_EPOCHS):
        epoch_fin += 1

        print(f'Epoch: {epoch+1} with early stopping counter in {counter} \n')
        model.train()
        avg_loss= train_epoch(DEVICE, train_loader, model, optimizer, loss_fn, scaler)
        #check acc - training set
        train_acc = compute_iou(train_loader, model, device=DEVICE)
        model.loss_acc['train_acc'].append(np.array(train_acc)) 
        
        running_vloss = 0.0

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(val_loader):
                vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels.long())
                running_vloss += vloss.item() * vinputs.size(0) 

        avg_vloss = running_vloss / len(val_loader.dataset)
        model.loss_acc["val_loss"].append(np.array(avg_vloss))

        #check acc - validation set
        val_acc = compute_iou(val_loader, model, device=DEVICE)
        model.loss_acc['val_acc'].append(np.array(val_acc))

        print('Training loss: {} \n Validation loss: {} \n'.format(avg_loss, avg_vloss))
        print('Training acc: {} \n Validation acc: {} \n'.format(train_acc, val_acc))
        
        save_loss(model=model, name='./mos2_unet_results/MoS2_unet_loss_acc')
    
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            counter = 0
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, name='./mos2_unet_results/MoS2_unet_best_model_checkpoint.pth.tar')
        else:
            counter += 1
        if counter > patience:
            print('Model stopped at {} epochs.'.format(epoch))
            break
        
    if epoch_fin == NUM_EPOCHS:
        print('Model finished all epochs!')

if __name__ == "__main__":
    train()
    print('Total Training Time: %.2f min' % ((time() - start)/60))
