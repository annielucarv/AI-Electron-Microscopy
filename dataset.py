import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Type
import warnings

## check_dim and norm_image
def check_image_dims(images: np.ndarray,
                     labels: np.ndarray,
                     num_classes: int
                     ):
    """
    Adds/remove if necessary pseudo-dimension of 1 (channel dimensions)
    to images and masks
    """
    if images.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training images',
            UserWarning)
        images = images[:, np.newaxis]
    
    if num_classes == 1 and labels.ndim == 3:
        warnings.warn(
            'Adding a channel dimension of 1 to training labels',
            UserWarning)
        labels = labels[:, np.newaxis]
    
    if num_classes > 1 and labels.ndim == 4:
        warnings.warn(
            'Removing the channel dimension from training labels',
            UserWarning)
        labels = labels.squeeze()

    return images, labels

def norm_image(image_data):

    """Normalize the images"""
    
    if image_data.ndim not in [3, 4]:
        raise AssertionError(
            "Provide image(s) as 3D (n, h, w) or 4D (n, 1, h, w) tensor")
    
    image_data = (image_data - image_data.min()) / np.ptp(image_data)

    return image_data

def numpy_to_torch(images: np.ndarray,
                labels: np.ndarray,
                num_classes: int
                ):
    
    """
    Transform dataset from numpy to torch
    """
    
    # normalizacao das imagens do dataset [0.,1.]
    

    # deixando o dataset no formato torch
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)


    images = images.float()
    
    if num_classes > 1: # necessario para cross-entropy
        labels = labels.long()
    else:
        labels = labels.float() 
    
    return images, labels

def get_loaders(images: np.ndarray, 
                labels: np.ndarray, 
                num_classes: int, 
                split: bool = True,
                # images_test: np.ndarray,
                # labels_test: np.ndarray, 
                batch_size: int = 32, 
                num_workers: int = 4, 
                pin_memory: bool = True):
                
    """
    Preprocess the training and test data, split the training data into training and validation the inicialize 3 PyTorch dataloader, train, validation and test
    """

    images = norm_image(images)

    images, labels = check_image_dims(images, labels, num_classes)

    if split:
    
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=256, random_state=42)
        # ('pre-numpy-to-torch')

        # print(f'X_train: {X_train.shape} \n X_val: {X_val.shape} \n y_train: {y_train.shape} \n y_val: {y_val.shape}')

        validation = numpy_to_torch(X_val, y_val, num_classes)

        (X_val, y_val) = validation

        # print('pos-numpy-to-torch')

        # print(f'X_train: {X_train.shape} \n X_val: {X_val.shape} \n y_train: {y_train.shape} \n y_val: {y_val.shape}')

        
        val_ds = torch.utils.data.TensorDataset(X_val, y_val)

        val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        )
    else:
        X_train, y_train = images, labels
    
    X_train, y_train = numpy_to_torch(X_train, y_train, num_classes)

    

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    
    # test_ds = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    if split:
        return train_loader, val_loader
    else:
        return train_loader
    
def get_flat_loaders(images: np.ndarray, 
                labels: np.ndarray, 
                num_classes: int, 
                split: bool = True,
                patch_size=4,
                # images_test: np.ndarray,
                # labels_test: np.ndarray, 
                batch_size: int = 32, 
                num_workers: int = 4, 
                pin_memory: bool = True):
    
    images = norm_image(images)

    if split:
    
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=.2, random_state=42)
        

        vimg_flat = torch.from_numpy(X_val.squeeze(1)).float().view(-1,1)
        vlabels_flat = torch.from_numpy(y_val.squeeze(1)).float().view(-1,1)


        val_dataset = torch.utils.data.TensorDataset(vimg_flat, vlabels_flat)

        val_loader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory, 
                                                 shuffle=False)

    else:
        X_train, y_train = images, labels


    img_flat = torch.from_numpy(X_train.squeeze(1)).float().view(-1,1)
    labels_flat = torch.from_numpy(y_train.squeeze(1)).float().view(-1,1)


    train_ds = torch.utils.data.TensorDataset(img_flat, labels_flat)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    if split:
        return train_loader, val_loader
    else:
        return train_loader
    
