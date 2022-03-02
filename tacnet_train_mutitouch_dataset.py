"""
Author: Quan Khanh Luu (JAIST)
Contact: quan-luu@jaist.ac.jp
Descriptions: This scripts train TacNet to reconstruct the skin shape on multi-point touch dataset (single + double touch/contact)
- Dataset: simulation dataset of single + double touch/contact
    Train dataset: 70% single touch + 70% double touch datasets
    Evaluation dataset: 30% single touch + 30% double touch datasets
- Input: Original pair of tactile images (6, 480, 640)
- Ouput: the nodal displacement vectors of free nodes of the artificial skin.
"""

import os

import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from dataset.tactile_image_dataset import TactileImageDataset
from networks.unet_model import TacNetUNet2

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('The hardware device used: {0}'.format(dev))

root_dir = '/home/protac/remoteDir'
init_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/init_data.csv')
single_label_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/single_point_touch')
double_label_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/output_label/double_point_touch')
single_input_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/tactile_image/single_point_touch')
double_input_path = os.path.join(root_dir, 'iotouch_env/train_data/simulated_data/tactile_image/double_point_touch')

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def get_model(lr, in_nc=6):
    model = TacNetUNet2(in_nc=in_nc)
    model.to(dev)
    # opt = optim.Adam(model.parameters(), lr=lr, betas=(0.04, 0.999))
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, opt

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb).float(), yb.float())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0 # loss per each batch
        batch = 0
        for xb, yb, _ in train_dl:
            batch+=1
            lossb, numb = loss_batch(model, loss_func, xb, yb, opt=opt)
            running_loss += lossb*numb
            if (batch)%50 == 0:
                print('[{}/{}][{}/{}]\tLoss : {}'.format(epoch+1, epochs, batch, len(train_dl), lossb))

        train_loss = running_loss/len(train_ds)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb, _ in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print('[{0}/{1}]\tTraining loss : {2}\tValidation loss : {3}'.format(epoch+1, epochs, train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses

def preprocess(x, y):
    return x.to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            xb, yb = self.func(b['images'], b['displacements'])
            yield (xb, yb, b['image_name'])

transform = transforms.Compose([
      transforms.CenterCrop(480),
      transforms.Resize((256,256)),
      transforms.ToTensor(),
])

train_single_label_path = os.path.join(single_label_path, 'train', 'single_label_pos.csv')
train_double_label_path = os.path.join(double_label_path, 'train', 'double_label_pos.csv')

train_single_input_path = os.path.join(single_input_path, 'train')
train_double_input_path = os.path.join(double_input_path, 'train')

single_train_ds = TactileImageDataset(init_path, train_single_label_path, train_single_input_path, transform=transform)
double_train_ds = TactileImageDataset(init_path, train_double_label_path, train_double_input_path, transform=transform)
print("The number of train single and double tactile dateset: {0}-{1}".format(len(single_train_ds), len(double_train_ds)))


test_single_label_path = os.path.join(single_label_path, 'test', 'single_label_pos.csv')
test_double_label_path = os.path.join(double_label_path, 'test', 'double_label_pos.csv')

test_single_input_path = os.path.join(single_input_path, 'test')
test_double_input_path = os.path.join(double_input_path, 'test')

single_test_ds = TactileImageDataset(init_path, test_single_label_path, test_single_input_path, transform=transform)
double_test_ds = TactileImageDataset(init_path, test_double_label_path, test_double_input_path, transform=transform)
print("The number of test single and double tactile dateset: {0}-{1}".format(len(single_test_ds), len(double_test_ds)))

train_ds = ConcatDataset([single_train_ds, double_train_ds])
print("The total number of train tactile dateset: {0}".format(len(train_ds)))
test_ds = ConcatDataset([single_test_ds, double_test_ds])
print("The total number of test tactile dateset: {0}".format(len(test_ds)))

bs = 32 # the number of tactile image paris in one batch
train_dl, valid_dl = get_data(train_ds, test_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# Check the batch size of the train_dl, valid_dl
xb, yb, _ = next(iter(train_dl))
print("Image batch shape : {}".format(xb.shape))
print("Labels batch shape :  {}".format(yb.shape))

# Get the tactile model
learning_rate = 0.015
model, opt = get_model(learning_rate)
model

# Check the weights of the parameters and the number of parameters in the constructed model
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Num of Element: {param.numel()}")

epochs = 30 # the number of iteration over a training dataset
# Define the loss function
mse_loss = nn.MSELoss()

# fit the model
train_losses, valid_losses = fit(epochs, model, mse_loss, opt, train_dl, valid_dl)

plt.plot(train_losses, 'r', label='train loss') # plotting t, a separately 
plt.plot(valid_losses, 'b', label='valid loss') # plotting t, b separately 
plt.legend()
figname = 'training_curve_TacNetUnet2_2022_01_25_single_double_touch_dataset.jpg'
plt.savefig(os.path.join(root_dir, 'iotouch_env/resources/training_curve', figname))

model_name = 'TacNetUnet2_2022_01_25_single_double_touch_dataset.pt' # name_year_month_day
model_name_path = os.path.join('iotouch_env/train_model', model_name)
model_dir = os.path.join(root_dir, model_name_path)
torch.save(model.state_dict(), model_dir)
