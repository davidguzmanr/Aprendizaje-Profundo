import os

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MyDataset(Dataset):
    """
    Dataset for the images in 'NIPS 2017: Non-targeted Adversarial Attack competition'.
    """
    
    def __init__(self, root_dir, images_csv, categories_csv, normalize=True):
        """
        Parameters
        ----------
        root_dir: str.
            Directory with all the images.
        
        images_csv: str.
            Path to the csv file with annotations of the images.
        
        categories_csv: str.
            Path to the csv file with the CategoryIds and CategoryNames.
            
        normalize: bool, default=True.
            If True apply transforms.Normalize with the mean and std from ImageNet.

        Returns
        -------
        Custom dataset

        Notes
        -----
        - Ver https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt para los nombres de
          las categorías que maneja PyTorch para ImageNet, el orden es el mismo que el de esta competencia.

        - Ver https://pytorch.org/tutorials/beginner/data_loading_tutorial.html para un tutorial.

        - Torchattacks only supports images with a range between 0 and 1, so DO NOT USE normalization transforms
          when making the attacks, create another dataset.
        """
        
        super(MyDataset, self).__init__()
        
        self.root_dir = root_dir
        self.annotations_df = pd.read_csv(images_csv)
        
        if normalize:
            self.transform = transforms.Compose([transforms.Resize(256), 
                                                 transforms.CenterCrop(224), 
                                                 transforms.ToTensor(), # [0,255] -> [0,1]
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 	                  std=[0.229, 0.224, 0.225])])
        else:
        	self.transform = transforms.Compose([transforms.Resize(256), 
                                                 transforms.CenterCrop(224), 
                                                 transforms.ToTensor()])


            
        categories_df = pd.read_csv(categories_csv)
        # Restamos uno para que los ids vayan de 0-999 y no de 1-1000
        ids = categories_df['CategoryId'].values - 1
        names = categories_df['CategoryName']        
        
        self.idx_to_class = dict(zip(ids, names))
        self.class_to_idx = dict(zip(names, ids))
            
    def __len__(self):
        return len(self.annotations_df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        img_name = os.path.join(self.root_dir, self.annotations_df.loc[idx, 'ImageId']) + '.png'
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)
        
        # La numeración de las clases en el csv de la competencia va de 1-1000, por lo que debemos 
        # restarle uno para que vaya de 0-999 y concuerde con el resto de los modelos.
        label = self.annotations_df.loc[idx, 'TrueLabel'] - 1
         
        return img, label