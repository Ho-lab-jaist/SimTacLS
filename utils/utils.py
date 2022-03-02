import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageOps

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def imshow(img, title=''):
    """Plot the image batch."""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

def img2tensor(img):
    """
    Convert image (cv2) to tensor array
    img: (H, W, C)
    tensor: (C, H, W)
    """
    img = (img/255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(img))

def tensor2img(tensor):
    if isinstance(tensor, torch.Tensor):  # get the data from a variable
        image_tensor = tensor.data

    img_numpy = image_tensor[0].cpu().float().numpy()
    img_numpy = (np.transpose(img_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img_numpy.astype(np.uint8)   


def apply_transform(img, method=Image.BICUBIC):
    if isinstance(img, np.ndarray):
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(480),
        transforms.Resize((256,256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_list = []
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, 256, 256, method)))
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list)

    return transform(img).unsqueeze(0)

def alpha_blending(background, foreground, alpha_mask, mask_inverted = True):    
    """Alpha blending for creating see-through-skin effect"""
    foreground = foreground.astype(float) # skin effect
    background = background.astype(float) # target pure scene

    # expand dimensions of alpha mask for multiplication operator
    alpha_mask = np.transpose(np.resize(alpha_mask, (3, alpha_mask.shape[0], alpha_mask.shape[1])), (1,2,0))
    
    if mask_inverted:
        alpha = 1.0 - (alpha_mask/255)
        foreground = np.multiply(alpha, foreground)
        background = np.multiply(1.0 - alpha, background)
    else:
        alpha = alpha_mask/255
        foreground = np.multiply(alpha, foreground)
        background = np.multiply(1.0 - alpha, background)

    outImage = np.clip(np.add(foreground, background), 0, 255)

    return outImage.astype('uint8')