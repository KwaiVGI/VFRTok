import cv2
import pandas as pd
from datasets import Dataset as HubDataset
from PIL import Image
from torch.utils.data import Dataset
import torch
import json
import redis
import random
import decord
from torchvision import transforms
from einops import rearrange
import random
import numpy as np

class RedisDataFrame:
    def __init__(self, key, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.key = key
        header_json = self.redis_client.lindex(key, 0)
        if header_json is None:
            raise ValueError(f"Cannot find {key} in redis.")
        self.columns = json.loads(header_json)
    
    def __getitem__(self, idx):
        # Redis 中第 0 行是表头，因此数据行索引需要偏移 +1
        row_json = self.redis_client.lindex(self.key, idx + 1)
        if row_json is None:
            raise IndexError("Index out of range.")
        return json.loads(row_json)
    
    def __len__(self):
        total = self.redis_client.llen(self.key)
        return total - 1 if total > 0 else 0

class ImageDataset(Dataset):
    def __init__(self, csv_file, data_column, transform):
        if csv_file.endswith('.csv'):
            # self.data_frame = pd.read_csv(csv_file)
            self.data_frame = HubDataset.from_csv(csv_file, cache_dir='/group/cache/datasets')
        else:
            self.data_frame = RedisDataFrame(csv_file)
        self.data_column = data_column
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # row = self.data_frame.iloc[idx]
        row = self.data_frame[idx]
        image_path = row[self.data_column]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image

def center_crop_tensor(tensor, image_size):
    _, _, h, w = tensor.shape
    assert h == image_size or w == image_size
    dx, dy = (w - image_size) // 2, (h - image_size) // 2
    return tensor[:, :, dy:dy+image_size, dx:dx+image_size]

class VideoDataset(Dataset):
    def __init__(self, csv_file, data_column, image_size, num_frames, fps=24, is_train=True, use_gpu=False, return_idx=False):
        self.data_frame = HubDataset.from_csv(csv_file, cache_dir='/group/cache/datasets')
        self.data_column = data_column

        self.image_size = image_size
        self.num_frames = num_frames
        self.fps = fps
        self.use_gpu = use_gpu
        self.is_train = is_train
        self.return_idx = return_idx
        decord.bridge.set_bridge("torch")

        transform = [
            transforms.Lambda(lambda x: rearrange(x, 'f h w c -> f c h w')),
            transforms.Lambda(lambda x: x.float() / 255.),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            transforms.Lambda(lambda x: center_crop_tensor(x, 256)),
        ]
        if is_train: 
            transform += [transforms.RandomHorizontalFlip()]
        transform += [transforms.Lambda(lambda x: rearrange(x, 'f c h w -> c f h w'))]

        self.transform = transforms.Compose(transform)
        
        assert not self.use_gpu # FIXME: not impl


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame[idx]
        video_path = row[self.data_column]
        h, w = self.get_read_size(row['height'], row['width'], self.image_size)
        frames = self.decord_read_video(video_path, h, w)

        if frames is None: # skip sample
            next_idx = (idx + 1) % self.__len__()
            return self.__getitem__(next_idx)

        frames = self.transform(frames)

        if self.return_idx:
            return frames, idx
        else:
            return frames
        
    def get_read_size(self, height, width, image_size):
        if height > width:
            height = int(height / width * image_size)
            width = image_size
        else:
            width = int(width / height * image_size)
            height = image_size
        return height, width

    def decord_read_video(self, video_path, height, width):
        if self.use_gpu:
            ctx = decord.gpu(torch.distributed.get_rank() % torch.cuda.device_count())
        else:
            ctx = decord.cpu(0)
        try:
            vr = decord.VideoReader(video_path, ctx=ctx, height=height, width=width)
            fps = vr.get_avg_fps()
            if fps < self.fps * 0.75:
                print(f'FPS of {video_path} too low ({fps}<{self.fps}*0.75)')
                return None
            num_samples = int(len(vr) / fps * self.fps)
            resample_indices = np.linspace(
                0, len(vr) - 1, num_samples
            ).astype(int)
            if num_samples < self.num_frames:
                print(f'No enough frames in {video_path} ({len(vr)}({fps})->{num_samples}({self.fps}) < {self.num_frames})')
                return None
            if self.is_train:
                start = random.randint(0, num_samples-1-self.num_frames)
            else:
                start = 0
            indices = [resample_indices[x] for x in range(start, start+self.num_frames)]
            frames = vr.get_batch(indices)
        except Exception as e:
            print(f'Failed to load {video_path} ({e})')
            return None
        return frames
    
if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = VideoDataset('../datasets/k600-train-256px.csv', 'video_path', 256, 16, 24, is_train=True)
    loader = DataLoader(dataset, batch_size=32, num_workers=32, shuffle=False)
    for _ in tqdm(loader):
        pass