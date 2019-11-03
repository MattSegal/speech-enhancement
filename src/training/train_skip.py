"""
Train SkipNet for super-resolution task using Deep Image Prior technique.
"""
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..models.images.skip_net import SkipNet

NUM_EPOCHS = 10
NOISE_FACTOR = 0.03
NOISE_CHANNELS = 32
IMAGE_PATH = "data/deep-image-prior/zebra.png"
SIZE_FACTOR = 4
LEARNING_RATE = 1e-2


def train(*args, **kwargs):
    net = SkipNet()
    criterion = nn.MSELoss()
    downsample = Downsampler(factor=SIZE_FACTOR)

    # Load low res image
    low_res_img = load_image()
    # (1, 3, 96, 144)

    # Create uniform noise
    img_h = SIZE_FACTOR * low_res_img.shape[2]
    img_w = SIZE_FACTOR * low_res_img.shape[3]
    inputs_shape = [1, NOISE_CHANNELS, img_h, img_w]
    inputs = torch.zeros(inputs_shape).uniform_() * NOISE_FACTOR

    # Initialize optmizer
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    for _ in tqdm(range(NUM_EPOCHS)):
        optimizer.zero_grad()
        # (1, 32, 384, 576)
        gen_image_high_res = net(inputs)
        # (1, 3, 384, 576)
        gen_image_low_res = downsample(gen_image_high_res)
        # (1, 3, 96, 144)
        loss = criterion(gen_image_low_res, low_res_img)
        loss.backward()
        optimizer.step()

    save_image_tensor(gen_image_high_res, "restored")
    save_image_tensor(gen_image_low_res, "restored_low_res")


def save_image_tensor(image_t, name):
    # Tensor has shape (1, C, H, W) rearrange to (H, W, C)
    # and convert floats 0 to 1 into ints 0 to 255
    image_arr = image_t.squeeze(dim=0).permute([1, 2, 0]).detach().numpy()
    image_arr_int = np.clip(image_arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_arr_int)
    img.save(IMAGE_PATH.replace("zebra", name))


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
    img_cropped.save(IMAGE_PATH.replace("zebra", "cropped"))

    low_res_size = [img_cropped.size[0] // SIZE_FACTOR, img_cropped.size[1] // SIZE_FACTOR]
    img_low_res = img_cropped.resize(low_res_size, Image.ANTIALIAS)
    # Save low res image
    img_low_res.save(IMAGE_PATH.replace("zebra", "low_res"))

    # Image has shape (H, W, C), rearrange to (1, C, H, W) and
    # turn ints 0 to 255 to floats 0 to 1
    img_arr = np.asarray(img_low_res).astype(np.float32) / 255.0
    img_t = torch.tensor(img_arr).permute(2, 0, 1)
    return img_t.reshape(1, *img_t.shape).detach()


class Downsampler(nn.Module):
    def __init__(self, factor=4):
        super(Downsampler, self).__init__()
        n_planes = 3
        support = 2
        phase = 0.5
        kernel_width = 4 * factor + 1
        kernel_type_ = "lanczos"
        kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support)
        downsampler = nn.Conv2d(
            n_planes, n_planes, kernel_size=kernel.shape, stride=factor, padding=0
        )
        self._downsampler = downsampler
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0
        kernel_torch = torch.from_numpy(kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch

        if kernel.shape[0] % 2 == 1:
            pad = int((kernel.shape[0] - 1) / 2.0)
        else:
            pad = int((kernel.shape[0] - factor) / 2.0)

        self.padding = nn.ReplicationPad2d(pad)

    def forward(self, input_t):
        return self._downsampler(self.padding(input_t))


def get_kernel(factor, kernel_type, phase, kernel_width, support):
    kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    center = (kernel_width + 1) / 2.0

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):

            if phase == 0.5:
                di = abs(i + 0.5 - center) / factor
                dj = abs(j + 0.5 - center) / factor
            else:
                di = abs(i - center) / factor
                dj = abs(j - center) / factor

            pi_sq = np.pi * np.pi

            val = 1
            if di != 0:
                val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                val = val / (np.pi * np.pi * di * di)

            if dj != 0:
                val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                val = val / (np.pi * np.pi * dj * dj)

            kernel[i - 1][j - 1] = val

    kernel /= kernel.sum()

    return kernel

