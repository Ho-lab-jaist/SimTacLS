import pandas as pd
import os
import glob

import torch
from torch.utils.data import Dataset

from PIL import Image


class TactileImageDataset(Dataset):
    def __init__(self, init_file, label_file, img_dir, transform=None, target_transform=None):
        self.labels_csv = pd.read_csv(label_file)
        self.labels_csv.set_index('ID', inplace = True)

        self.init_pos_csv = pd.read_csv(init_file)
        self.transform = transform
        self.target_transform = target_transform

        # get image paths
        self.top_img_paths = sorted(glob.glob(img_dir+'/up/*.jpg'))
        self.bot_img_paths = sorted(glob.glob(img_dir+'/bot/*.jpg'))
        assert len(self.top_img_paths)==len(self.bot_img_paths), 'Number of images between cameras mismatched'

        # get initial/non-deformed node positions
        self.init_pos = torch.tensor(self.init_pos_csv.iloc[0, 1:], dtype=float)

    def __len__(self):
        return len(self.top_img_paths)

    def __getitem__(self, idx):
        top_img_path = self.top_img_paths[idx]
        bot_img_path = self.bot_img_paths[idx]
        img_name = os.path.basename(top_img_path)
        # read image in the given directory, and convert the images (.jpg format) into tensors
        image_up = Image.open(top_img_path).convert('RGB') # read the respective upper image
        image_bot = Image.open(bot_img_path).convert('RGB') # read the respective bottom image
        # get the training labels from the .csv file
        label = (torch.tensor(self.labels_csv.loc[img_name], dtype=float) - self.init_pos)
        if self.transform:
            image_up = self.transform(image_up)
            image_bot = self.transform(image_bot)
        if self.target_transform:
            label = self.target_transform(label)
        # concatenate the two tactile images
        tactile_image = torch.cat((image_up, image_bot), dim=0)
        return {'images': tactile_image, 'displacements': label, 'image_name': img_name}

class TactileImageDataset2(Dataset):
    def __init__(self, init_file, label_file, img_dir, transform=None, target_transform=None):
        self.labels_csv = pd.read_csv(label_file)
        self.labels_csv.set_index('ID', inplace = True)

        self.init_pos_csv = pd.read_csv(init_file)
        self.transform = transform
        self.target_transform = target_transform

        # get image paths
        self.top_img_paths = sorted(glob.glob(img_dir+'/up/*.jpg'))
        self.bot_img_paths = sorted(glob.glob(img_dir+'/bot/*.jpg'))
        assert len(self.top_img_paths)==len(self.bot_img_paths), 'Number of images between cameras mismatched'

        # get initial/non-deformed node positions
        self.init_pos = torch.tensor(self.init_pos_csv.iloc[0, 1:], dtype=float)

    def __len__(self):
        return len(self.top_img_paths)

    def __getitem__(self, idx):
        top_img_path = self.top_img_paths[idx]
        bot_img_path = self.bot_img_paths[idx]
        img_name = os.path.basename(top_img_path)
        # read image in the given directory, and convert the images (.jpg format) into tensors
        image_up = Image.open(top_img_path).convert('RGB') # read the respective upper image
        image_bot = Image.open(bot_img_path).convert('RGB') # read the respective bottom image
        # get the training labels from the .csv file
        label_temp = (torch.tensor(self.labels_csv.loc[img_name], dtype=float) - self.init_pos).reshape(-1, 3)
        # norm_displacements = torch.linalg.vector_norm(label_temp, axis=1)
        label = torch.cat((label_temp[:, 0].unsqueeze(0), label_temp[:, 1].unsqueeze(0), label_temp[:, 2].unsqueeze(0)), dim=0)
        if self.transform:
            image_up = self.transform(image_up)
            image_bot = self.transform(image_bot)
        if self.target_transform:
            label = self.target_transform(label)
        # concatenate the two tactile images
        tactile_image = torch.cat((image_up, image_bot), dim=0)
        return {'images': tactile_image, 'displacements': label, 'image_name': img_name}

if __name__=='__main__':
    
    root_dir = '/home/protac/remoteDir'
    init_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/init_data.csv')
    single_label_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/single_point_touch/label_pos.csv')
    double_label_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/double_point_touch/label_pos.csv')
    single_input_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/tactile_image/single_point_touch')
    double_input_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/tactile_image/double_point_touch')

    from torchvision import transforms
    from torch.utils.data import random_split
    from torch.utils.data import ConcatDataset

    transform = transforms.Compose([
        transforms.CenterCrop(480),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    single_tactile_ds = TactileImageDataset(init_path, single_label_path, single_input_path, transform=transform)
    double_tactile_ds = TactileImageDataset(init_path, double_label_path, double_input_path, transform=transform)
    print("The total number of tactile dateset: {0}-{1}".format(len(single_tactile_ds), len(double_tactile_ds)))

    single_train_ds, single_valid_ds = random_split(single_tactile_ds, [int(0.8*len(single_tactile_ds)), int(0.2*len(single_tactile_ds))])
    double_train_ds, double_valid_ds = random_split(double_tactile_ds, [int(0.8*len(double_tactile_ds)), int(0.2*len(double_tactile_ds))])
    train_ds = ConcatDataset([single_train_ds, double_train_ds])
    valid_ds = ConcatDataset([single_valid_ds, double_valid_ds])
    print("The number of train and valid dateset : {0}-{1}".format(len(train_ds), len(valid_ds)))