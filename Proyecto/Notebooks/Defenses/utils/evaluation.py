import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NormalizationLayer(nn.Module) :
    """
    Normalization layer to add the beginning of the models.
    """
    def __init__(self, mean, std) :
        super(NormalizationLayer, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def get_topk_accuracy(model, dataloader):
    """
    Computes the accuracy@1 and accuracy@5 of the model in the dataloader.

    Parameters
    ----------
    model: callable
        PyTorch model.
    dataloader: dataloader with the images, batch_size must be None.
        PyTorch DataLoader.

    Returns
    -------
    Tuple with (y_true, y_pred, accuracy@1, accuracy@5).

    Notes
    -----

    """

    model.eval()

    y_true = []
    y_pred_top1 = []
    y_pred_top5 = []

    for img, label in tqdm(dataloader):
        output = model(img.unsqueeze(0).to(device))
        probabilities = F.softmax(output[0], dim=0)
        
        top1_prob, top1_id = torch.topk(probabilities, k=1)
        top5_prob, top5_id = torch.topk(probabilities, k=5)
        
        y_true.append(label)
        y_pred_top1.append(top1_id.tolist())
        y_pred_top5.append(top5_id.tolist())

    accuracy_1 = np.array([1 if label in prediction else 0 for (label, prediction) in zip(y_true, y_pred_top1)]).mean()
    accuracy_5 = np.array([1 if label in prediction else 0 for (label, prediction) in zip(y_true, y_pred_top5)]).mean()

    return y_true, y_pred_top1, 100*accuracy_1, 100*accuracy_5


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        # The clone is important, because torchvision.transforms.Normalize's operations are in-place 
        return super().__call__(tensor.clone())


def plot_adversarial(dataset, adv_dataset, y_pred, y_pred_adv, i):
    """
    Plot the i-th original image and the i-th adversarial image.

    Parameters
    ----------
    dataset: PyTorch dataset
        Dataset with the original images.

    adv_dataset: PyTorch dataset
        Dataset with the adversarial images.

    y_pred: array
        Array with the predictions of the model in dataset.

    y_pred_adv: array
        Array with the predictions of the model in adv_dataset.

    i: int
        Index of the image.

    Returns
    -------
    None

    Notes
    -----

    """

    img, _ = dataset.__getitem__(i)
    adv_img, _ = adv_dataset.__getitem__(i)

    noise = (adv_img - img)
    noise = noise.permute(1,2,0).numpy()
    noise = noise/np.max(noise)
    noise = np.clip(noise, 0, 1)

    # Dictionary with the index of each class
    idx_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    fig, (ax0, ax1, ax2) = plt.subplots(figsize=(15,6), nrows=1, ncols=3)

    ax0.imshow(img.permute(1,2,0).numpy())
    ax0.set_title(idx_to_class[y_pred[i][0]].split(',')[0].capitalize())

    ax1.imshow(adv_img.permute(1,2,0).numpy())
    ax1.set_title(idx_to_class[y_pred_adv[i][0]].split(',')[0].capitalize())

    ax2.imshow(noise)
    ax2.set_title('Noise')

    plt.show()

def get_same_predictions(y_pred, y_pred_adv):
    """
    Get the indexes of the predictions where the image and the adversarial image where
    classified as the same class.

    Parameters
    ----------
    y_pred: array
        Array with the predictions of the model in dataset.

    y_pred_adv: array
        Array with the predictions of the model in adv_dataset.

    Returns
    -------
    Array.
    """

    indexes = [i for i, (y, y_adv) in enumerate(zip(y_pred, y_pred_adv)) if y == y_adv]

    return indexes

def get_different_predictions(y_pred, y_pred_adv):
    """
    Get the indexes of the predictions where the image and the adversarial image where
    classified differently.

    Parameters
    ----------
    y_pred: array
        Array with the predictions of the model in dataset.

    y_pred_adv: array
        Array with the predictions of the model in adv_dataset.

    Returns
    -------
    Array.
    """

    indexes = [i for i, (y, y_adv) in enumerate(zip(y_pred, y_pred_adv)) if y != y_adv]

    return indexes

