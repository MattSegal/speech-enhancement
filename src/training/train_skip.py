"""
Train SkipNet for super-resolution task using Deep Image Prior technique.
"""
import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..models.images.skip_net import SkipNet

NUM_EPOCHS = 2000
NOISE_REG = 0.03
NOISE_CHANNELS = 32
IMAGE_NAME = "zebra"
IMAGE_EXT = "png"
IMAGE_PATH = f"data/deep-image-prior/{IMAGE_NAME}/original.{IMAGE_EXT}"
SIZE_FACTOR = 4
LEARNING_RATE = 1e-2


def train(*args, **kwargs):
    net = SkipNet().cuda()
    criterion = nn.MSELoss()

    # Load low res image
    low_res_img = load_image().cuda()

    # Create uniform noise
    img_h = SIZE_FACTOR * low_res_img.shape[2]
    img_w = SIZE_FACTOR * low_res_img.shape[3]
    inputs_shape = [1, NOISE_CHANNELS, img_h, img_w]
    inputs = torch.zeros(inputs_shape).detach().uniform_().cuda()
    noise = torch.zeros(inputs.shape).detach().cuda()

    # Initialize optmizer
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(NUM_EPOCHS)):
        # Pertubate input noise a little
        inputs_reg = inputs + noise.normal_() * NOISE_REG
        optimizer.zero_grad()
        gen_image_high_res = net(inputs_reg)
        gen_image_low_res = interpolate(
            gen_image_high_res, scale_factor=(1 / SIZE_FACTOR), mode="bilinear"
        )
        loss = criterion(gen_image_low_res, low_res_img)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            save_image_tensor(gen_image_high_res, "restored")
            save_image_tensor(gen_image_low_res, "restored_low_res")

    save_image_tensor(gen_image_high_res, "restored")
    save_image_tensor(gen_image_low_res, "restored_low_res")


def save_image_tensor(image_t, name):
    # Tensor has shape (1, C, H, W) rearrange to (H, W, C)
    # and convert floats 0 to 1 into ints 0 to 255
    image_arr = image_t.squeeze(dim=0).permute([1, 2, 0]).cpu().detach().numpy()
    image_arr_int = np.clip(image_arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_arr_int)
    img.save(IMAGE_PATH.replace("original", name))


def load_image():
    # Load the original image from disk
    img = Image.open(IMAGE_PATH)
    # Crop image so that its dimensions are divisible by 2**5 (32)
    new_h = img.size[0] - img.size[0] % 32
    new_w = img.size[1] - img.size[1] % 32
    bounding_box = [
        (img.size[0] - new_h) / 2,
        (img.size[1] - new_w) / 2,
        (img.size[0] + new_h) / 2,
        (img.size[1] + new_w) / 2,
    ]
    img_cropped = img.crop(bounding_box)
    # Save cropped image
    img_cropped.save(IMAGE_PATH.replace("original", "cropped"))
    # Downsample image
    low_res_size = [
        img_cropped.size[0] // SIZE_FACTOR,
        img_cropped.size[1] // SIZE_FACTOR,
    ]
    img_low_res = img_cropped.resize(low_res_size, Image.ANTIALIAS)
    # Save low res image
    img_low_res.save(IMAGE_PATH.replace("original", "low_res"))
    # Image has shape (H, W, C), rearrange to (1, C, H, W) and
    # turn ints 0 to 255 to floats 0 to 1
    img_arr = np.asarray(img_low_res).astype(np.float32) / 255.0
    img_t = torch.tensor(img_arr).permute(2, 0, 1)
    return img_t.reshape(1, *img_t.shape).detach()
