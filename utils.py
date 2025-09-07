import os

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
import cv2
from skimage.metrics import peak_signal_noise_ratio
import logging
import numpy as np
from pathlib import Path
import random
image_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
class Normalize(nn.Module):
    def forward(self, x):
        print("cnm",x.shape)
        x = x.float() / 255.0
        n, f, c, h, w = x.shape
        return x.view(n, f * c, h, w)


class Augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_names = [
            'do_nothing', 'flipud', 'rot90', 'rot90_flipud',
            'rot180', 'rot180_flipud', 'rot270', 'rot270_flipud', 'add_noise'
        ]
        self.weights = [32, 12, 12, 12, 12, 12, 12, 12, 12]

    def augment(self, op_name, img):

        match op_name:
            case 'do_nothing':
                return img
            case 'flipud':
                return torch.flip(img, dims=[1])
            case 'rot90':
                return torch.rot90(img, k=1, dims=[1, 2])
            case 'rot90_flipud':
                return torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1])
            case 'rot180':
                return torch.rot90(img, k=2, dims=[1, 2])
            case 'rot180_flipud':
                return torch.flip(torch.rot90(img, k=2, dims=[1, 2]), dims=[1])
            case 'rot270':
                return torch.rot90(img, k=3, dims=[1, 2])
            case 'rot270_flipud':
                return torch.flip(torch.rot90(img, k=3, dims=[1, 2]), dims=[1])
            case 'add_noise':
                noise = torch.randn(1, 1, 1, device=img.device) * (5.0 / 255.0)
                return torch.clamp(img + noise.expand_as(img), 0.0, 1.0)
            case _:
                raise ValueError(f"Unsupported op name: {op_name}")

    def forward(self, x):
        N, FC, H, W = x.shape
        F = FC // 3
        out = torch.empty_like(x)
        op_name = random.choices(self.op_names, weights=self.weights, k=1)[0]
        print(op_name, "new")
        for n in range(N):
            for f in range(F):
                img = x[n, f * 3:f * 3 + 3]  # [3, H, W]
                out[n, f * 3:f * 3 + 3] = self.augment(op_name, img)
        return out


def normalize_augment(data_input, ctrl_fr_idx):
    video_transform = Compose([
        Normalize(),
        Augment(),
    ])
    img_train = video_transform(data_input)
    gt_train = img_train[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
    return img_train, gt_train


def get_image_names(seq_dir, pattern=None):
    seq_path = Path(seq_dir)
    files = []

    for image_type in image_types:
        files.extend(seq_path.glob(image_type))

    if pattern is not None:
        files = [file for file in files if pattern in file.name]

    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.name))))
    return [str(file) for file in files]


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    if gray_mode:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)

    if expand_axis0:
        img = np.expand_dims(img, axis=0)

    expanded_h, expanded_w = False, False
    h, w = img.shape[-2], img.shape[-1]

    if expand_if_needed:
        if h % 2 == 1:
            expanded_h = True
            last_row = img[..., -1:, :]
            img = np.concatenate((img, last_row), axis=-2)

        if w % 2 == 1:
            expanded_w = True
            last_col = img[..., -1:]  # slice last col
            img = np.concatenate((img, last_col), axis=-1)

    if normalize_data:
        img = img.astype(np.float32) / 255.0

    return img, expanded_h, expanded_w

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    file_paths = get_image_names(seq_dir)
    file_paths = file_paths[:max_num_fr]

    print(f"Open sequence in folder: {seq_dir} ({len(file_paths)} frames)")

    seq_list = []
    expanded_h, expanded_w = False, False

    for fpath in file_paths:
        img, h_exp, w_exp = open_image(
            fpath,
            gray_mode=gray_mode,
            expand_if_needed=expand_if_needed,
            expand_axis0=False
        )
        seq_list.append(img)
        expanded_h |= h_exp
        expanded_w |= w_exp

    # Stack to [T, C, H, W]
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w


def setup_logger(save_dir):
    """Setup logger for training"""
    log_file = os.path.join(save_dir, 'train.log')
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def batch_psnr(images, images_clean, data_range):
    images_cpu = images.data.cpu().numpy().astype(np.float32)
    images_clean = images_clean.data.cpu().numpy().astype(np.float32)
    psnr = 0.0
    for index in range(images_cpu.shape[0]):
        psnr += peak_signal_noise_ratio(images_clean[index, :, :, :], images_cpu[index, :, :, :],
                                        data_range=data_range)
    return psnr / images_cpu.shape[0]

def save_checkpoint(model, optimizer, schedular, epoch,current_step, save_dir):
    checkpoint = {
        'epoch': epoch,
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': schedular.state_dict(),
    }

    # Regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{current_step}.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
