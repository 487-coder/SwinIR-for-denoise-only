
import torch
import cv2,random
import numpy as np
import imageio.v3 as iio
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from utils import open_sequence
from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import Dataset
NUMFRXSEQ_VAL = 1
VALSEQPATT = '*'

class VideoSequenceDataset(Dataset):
    def __init__(self, file_root, sequence_length, crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
        """
        Args:
            file_root:        目录路径，包含若干 .mp4 视频
            sequence_length:  每个序列的帧数 F（如 5）
            crop_size:        每帧空间裁剪大小
            epoch_size:       每个 epoch 的样本数量（<0 表示用全集）
            random_shuffle:   是否随机取样中心帧
            temp_stride:      时间间隔（<0 表示等于 sequence_length）
            sequence_length = 1, 后续需要加噪声，做增强
        """
        super().__init__()
        video_paths = sorted(Path(file_root).glob("*.mp4"))
        assert video_paths, f"No .mp4 videos found in {file_root}"
        self.video_paths = video_paths

        # 预加载所有视频帧到内存
        self.videos = []
        for p in self.video_paths:
            frames = iio.imread(str(p))  # [T, H, W, 3], RGB
            self.videos.append(np.asarray(frames))

        self.seq_len = sequence_length
        self.half = (sequence_length - 1) // 2
        self.stride = temp_stride if temp_stride > 0 else sequence_length
        self.crop_size = crop_size
        self.random_shuffle = random_shuffle
        self.to_tensor = T.ToTensor()

        self.pairs = []
        for vid, frames in enumerate(self.videos):
            T_frames = frames.shape[0]
            valid_centers = list(range(
                self.half * self.stride,
                T_frames - self.half * self.stride
            ))
            for c in valid_centers:
                self.pairs.append((vid, c))

        self.epoch_size = epoch_size if epoch_size > 0 else len(self.pairs)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.random_shuffle:
            vid, center = random.choice(self.pairs)
        else:
            vid, center = self.pairs[idx % len(self.pairs)]

        video = self.videos[vid]  # [T, H, W, 3]

        seq_imgs = []
        for i in range(self.seq_len):
            frame_idx = center + (i - self.half) * self.stride
            arr = video[frame_idx]  # H,W,3 uint8
            t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0)
            seq_imgs.append(t)
        seq = torch.stack(seq_imgs, dim=0)

        i, j, h, w = T.RandomCrop.get_params(seq[0], output_size=(self.crop_size, self.crop_size))
        seq = seq[:, :, i:i + h, j:j + w]  # [F, C, Hc, Wc]
        # 中心帧 GT： [1, C, H, W]
        gt = seq[self.half]  # 取第 self.half 帧
        return seq, gt

class TestDataset(Dataset):

    def __init__(self, data_dir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
        self.gray_mode = gray_mode
        self.data_dir = Path(data_dir)
        self.num_input_frames = num_input_frames

        seq_dirs = sorted(self.data_dir.rglob(VALSEQPATT))
        seq_dirs = [p for p in seq_dirs if p.is_dir() and not p.name.startswith('.')]
        self.sequences = []
        for seq_path in seq_dirs:
            seq, _, _ = open_sequence(
                str(seq_path),  # open_sequence likely expects string path
                gray_mode=gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames
            )
            self.sequences.append(seq)

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)