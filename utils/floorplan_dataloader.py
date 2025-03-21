import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pickle
import numpy as np


MAX_NUM_NODES = 100
ENCODED_DIM = 3

class ImageDataset(Dataset):
    def __init__(self, image_folder, graph_folder=None, transform=None, img_size=256):
        """
        Custom dataset for loading PNG images from a folder.

        Args:
        - image_folder (str): Path to the folder containing PNG images.
        - transform (torchvision.transforms): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.graph_folder = graph_folder
        self.cond_image_files = [f for f in os.listdir(image_folder) if f.endswith('_cond.png')]
        # only load PNG images that have number.png format
        # self.image_files = [f for f in self.image_files if f.split('.')[0].isdigit()]
        self.image_files = [f.replace("_cond.png", ".png") for f in self.cond_image_files]
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        # resize image 
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.BICUBIC)
        if self.transform:
            image = self.transform(image)
       
        cond_img_path = os.path.join(self.image_folder, self.cond_image_files[idx])
        cond_image = Image.open(cond_img_path).convert("RGB")
        # resize image
        if cond_image.size != (self.img_size, self.img_size):
            cond_image = cond_image.resize((self.img_size, self.img_size), Image.BICUBIC)
        if self.transform:
            cond_image = self.transform(cond_image)

        return image, cond_image, 1, img_path

if __name__ == "__main__":
    # transform = transforms.Compose([
    #     # transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = transforms.Compose([
        transforms.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32)),  # Convert to tensor without scaling
        transforms.Lambda(lambda x: (x / 14.0) * 2.0 - 1.0)  # Min-Max scaling to [-1, 1]
    ])
    dataset = ImageDataset("/home/donaldtrump/haolan/msd_dataset/DIT_data_3/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (image, cond_image, _) in enumerate(dataloader):
        print(image.shape, cond_image.shape)
        if i == 0:
            break
    
    # do not shuffle the dataset
    dataset = ImageDataset("/home/donaldtrump/haolan/msd_dataset/DIT_data_3/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    