import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F

# from torchmetrics import JaccardIndex

def save_loss(model, name='model_loss_acc'):
    np.savez(name, train_loss = model.loss_acc["train_loss"], val_loss=model.loss_acc["val_loss"], train_acc= model.loss_acc["train_acc"], val_acc=model.loss_acc["val_acc"])

def save_checkpoint(state, name= 'model_checkpoint.pth.tar'):
    print('Saving checkpoint')
    torch.save(state, name)

def load_checkpoint(model, checkpoint):
    checkpoint = torch.load(checkpoint)
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # print(y.shape)
            inference = model(x)
            # print(inference.shape)
            if inference.shape[1] == 1:
                preds = torch.sigmoid(inference)
                preds = (preds > 0.5).float()
            else:
                preds = F.softmax(inference, dim = 1)
                # print(preds.shape)
                preds = torch.argmax(preds, 1)
                # print(preds.shape)
                y = y.squeeze()
                # print(y.shape)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )

    # print(
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    # print(f"Dice score: {dice_score/len(loader)}")
    return num_correct/num_pixels

def compute_iou(loader, model,device='cuda', num_classes=3, ignore_index=None):
    """
    Computes per-class IoU and mean IoU.

    Args:
        preds (Tensor): Raw logits from model, shape [B, C, H, W]
        labels (Tensor): Ground truth labels, shape [B, H, W]
        num_classes (int): Number of classes
        ignore_index (int, optional): Label index to ignore

    Returns:
        ious (Tensor): Per-class IoUs
        mean_iou (float): Mean IoU over valid classes
    """
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            labels = y.to(device)
            # print(y.shape)
            preds = model(x)
        preds = torch.argmax(preds, dim=1)
        ious = []
        for cls in range(num_classes):
            if ignore_index is not None and cls == ignore_index:
                continue

            pred_inds = (preds == cls)
            label_inds = (labels == cls)

            intersection = (pred_inds & label_inds).sum().float()
            union = (pred_inds | label_inds).sum().float()

            if union == 0:
                iou = torch.tensor(1.0) 
            else:
                iou = intersection / union

            ious.append(iou)

        ious_tensor = torch.tensor(ious)
        mean_iou = torch.nanmean(ious_tensor)  # handles possible NaNs
        return mean_iou.item()

def compute_binary_iou(loader, model, device='cuda', threshold=0.5):
    """
    Computes IoU for binary semantic segmentation.

    Args:
        loader (DataLoader): PyTorch DataLoader with (image, mask) batches
        model (torch.nn.Module): The model to evaluate
        device (str): 'cuda' or 'cpu'
        threshold (float): Threshold for converting probabilities to binary predictions

    Returns:
        iou (float): Intersection over Union for the foreground class
    """
    model.eval()
    intersection_total = 0.0
    union_total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # shape: [B, 1, H, W] or [B, H, W]
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)  # binary logits
                preds = (probs > threshold)
            else:
                # Handle if output is [B, 2, H, W] with logits for both classes
                preds = torch.argmax(logits, dim=1)

            intersection = ((preds == 1) & (y == 1)).sum().item()
            union = ((preds == 1) | (y == 1)).sum().item()

            intersection_total += intersection
            union_total += union

    if union_total == 0:
        return 1.0  # Perfect score if there are no positive samples
    else:
        return intersection_total / union_total

def predict(model, device, images: torch.Tensor, nb_classes= 3) -> np.ndarray:
        """
        Returns 'probability' of each pixel
        in image(s) belonging to an atom/defect
        """
        images = images.to(device)
        model.eval()
        with torch.no_grad():
            prob = model(images)
        
        if images.shape[1] ==1:
            prob = torch.sigmoid(prob)
        else:
            prob = F.softmax(prob, dim = 1)

        prob = prob.permute(0, 2, 3, 1)  # reshape to have channel as a last dim
        prob = prob.cpu()
        return prob

def prob_to_binary(outputs, to_torch = True):

    pred = np.zeros((outputs.shape[:3]))
    img_pixel = int(outputs.shape[2])
    for i in range(len(outputs)):
        new = outputs[i].reshape(1,img_pixel,img_pixel)
        new = np.rint(new)
        pred[i] = new
    if to_torch:
        pred = torch.from_numpy(pred)
        
    return pred

def check_pixel_accuracy(device, loader, model):
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.float().to(device)
            targets = targets.float().to(device)

            outputs = model(features)
            probs = torch.sigmoid(outputs)  # convert logits to probabilities

            preds = (probs >= 0.5).float()  # threshold at 0.5

            correct += (preds == targets).sum().item()
            total += targets.numel()

    accuracy = correct / total
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy