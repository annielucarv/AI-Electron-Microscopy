from model import MultilayerLinear
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from dataset import get_flat_loaders
from utils import *

start = time()

print('Linear Feedforward model for graphene dataset')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-3
NUM_EPOCHS = 500
BATCH_SIZE = 16

graphene_ds = np.load('./datasets/graphene_dataset.npz')

images, labels = graphene_ds['images'], graphene_ds['labels']

def train_epoch(device, loader, model, optimizer, loss_batch, scaler):
    running_loss = 0.

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.float().to(device)
        targets = targets.float().to(device)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(features)
            loss = loss_batch(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * features.size(0)

    avg_loss = running_loss / len(loader.dataset)
    model.loss_acc["train_loss"].append(np.array(avg_loss))
    return avg_loss

def train():
    best_vloss = float('inf')
    counter = 0
    patience = 70

    model = MultilayerLinear(num_classes=1)
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_flat_loaders(
        images=images[:1000], 
        labels=labels[:1000],
        split=True,
        num_classes=1,
        batch_size=1024,
        num_workers=4,
        pin_memory=True
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        avg_loss = train_epoch(DEVICE, train_loader, model, optimizer, loss_fn, scaler)
        
        # Accuracy on train
        train_acc = check_pixel_accuracy(DEVICE, train_loader, model)
        model.loss_acc['train_acc'].append(np.array(train_acc))

        # Validation loss
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                vinputs = vinputs.float().to(DEVICE)
                vlabels = vlabels.float().to(DEVICE)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item() * vinputs.size(0)

        avg_vloss = running_vloss / len(val_loader.dataset)
        model.loss_acc["val_loss"].append(np.array(avg_vloss))

        # Accuracy on validation
        val_acc = check_pixel_accuracy(loader=val_loader, model=model, device=DEVICE)
        model.loss_acc['val_acc'].append(np.array(val_acc))

        # Epoch summary
        print(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
        print(f'Patience: {patience}')
        print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")

        save_loss(model=model, name='./graphene_MLL_results/graphene_MLL_loss_acc')

        # Checkpoint
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            counter = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, name='./graphene_MLL_results/graphene_MLL_best_model_checkpoint.pth.tar')
        else:
            counter += 1

        if counter > patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("Training complete!")

if __name__ == "__main__":
    train()
    print('Total Training Time: %.2f min' % ((time() - start) / 60))

