import deepspeed
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from einops import rearrange
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
from loguru import logger
import argparse
import ruamel.yaml as yaml
import sys
import os, shutil
import cv2
import numpy as np
from modelling.tokenizer import Tokenizers
from utils.misc import load_model_state_dict
from torch.utils.data import DataLoader
from data.dataset import VideoDataset

def build_model(config, ckpt_path):
    model = Tokenizers[config['model']](
        image_size=config['image_size'],
        num_frames=config['num_frames'],
        codebook_embed_dim=config['codebook_embed_dim'],
        enc_type=config['enc_type'],
        encoder_model=config['encoder_model'],
        dec_type=config['dec_type'],
        decoder_model=config['decoder_model'],
        num_latent_tokens=config['num_latent_tokens'],
        enc_patch_size=config['encoder_patch_size'],
        dec_patch_size=config['decoder_patch_size'],
        t_patch_size=config['t_patch_size'],
        enc_pretrained=False,
        dec_pretrained=False,
        use_ape=config['use_ape'],
        use_rope=config['use_rope'],
        rope_mixed=config['rope_mixed'],
        rope_heads=config.get('rope_heads', None),
        rope_theta=config['rope_theta'],
        rope_theta_t=config.get('rope_theta_t', 100.0),
        variable_num_frames=config.get('variable_num_frames', False),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(load_model_state_dict(checkpoint), strict=False)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.config = config # set config
    return model

def get_num_frames(period, fps, t_patch_size):
    return int(round(period * fps / t_patch_size) * t_patch_size)

def build_recovery_transforms(norm_mean, norm_std):
    norm_mean = [-m / s for m, s in zip(norm_mean, norm_std)]
    norm_std = [1 / s for s in norm_std]
    return transforms.Compose([
        transforms.Lambda(lambda x: rearrange(x, 'b c f h w -> b f c h w')),
        transforms.Normalize(norm_mean, norm_std),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Lambda(lambda x: x.detach().cpu().float()),
    ]) # b c f h w -> b f c h w, range 0 to 1

def save_video(path: str, frames: torch.Tensor, fps: int):
    """
    should already processed by recovery_transforms
    """
    assert frames.ndim == 4
    f, c, h, w = frames.shape
    assert c == 3 # 1 3 h w
    assert 0 <= frames.min() and frames.max() <= 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    frames = rearrange(frames, 'f c h w -> f h w c')
    for frame in frames:
        frame = (frame.numpy() * 255.).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        config = file_yaml.load(f)
    
    # Setup DDP
    deepspeed.init_distributed()
    rank = deepspeed.comm.get_rank()
    if rank != 0:
        logger.remove() # remove all logs
        logger.add(sys.stderr, level="ERROR") # only show error on screen

    # infer config
    in_folder = os.path.join(args.results_dir, 'inputs')
    out_folder = os.path.join(args.results_dir, 'outputs')
    if rank == 0:
        if os.path.exists(args.results_dir):
            logger.warning(f'Output dir exsists, delete first!')
            shutil.rmtree(args.results_dir)
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(in_folder, exist_ok=True)
        os.makedirs(out_folder, exist_ok=True)
    deepspeed.comm.barrier()
    if rank == 0:
        logger.add(os.path.join(args.results_dir, "{time}.log"), enqueue=True)
    logger.info(f"Starting, world_size={deepspeed.comm.get_world_size()}.")
    logger.info(f'Output dir: {args.results_dir}')
    logger.info(f'Config: {config}')

    # create and load model
    model = build_model(config, args.ckpt)

    # Setup data
    enc_num_frames = get_num_frames(config['period'], args.enc_fps, config['t_patch_size'])
    dec_num_frames = get_num_frames(config['period'], args.dec_fps, config['t_patch_size'])
    dataset = VideoDataset(
        csv_file=args.csv_path, 
        data_column='video_path', 
        image_size=config['image_size'], 
        num_frames=enc_num_frames, 
        fps=args.enc_fps,
        is_train=False,
        return_idx=True
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=deepspeed.comm.get_world_size(),
        rank=deepspeed.comm.get_rank(),
        shuffle=False,
        seed=0
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    recovery_transforms = build_recovery_transforms([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # Init deepspeed engine
    torch.cuda.set_device(rank)
    model_engine = deepspeed.init_inference(model=model, dtype=torch.bfloat16)
    model_engine.eval()
    logger.info(f"Finish model deepspeed initialization.")

    dtype = next(model_engine.module.parameters()).dtype
    device = next(model_engine.module.parameters()).device

    deepspeed.comm.barrier()

    for batch in tqdm(loader, total=len(loader), disable=rank!=0):
        frames, idxs = batch
        frames = frames.to(device=device, dtype=dtype)

        with torch.no_grad():
            out = model_engine(frames, enc_num_frames, args.enc_fps, dec_num_frames, args.dec_fps)
        out_frames = out[0]

        frames = recovery_transforms(frames)
        out_frames = recovery_transforms(out_frames)
        for i, gt, pred in zip(idxs, frames, out_frames):
            gt_path = os.path.join(in_folder, f'{i:07d}.mp4')
            pred_path = os.path.join(out_folder, f'{i:07d}.mp4')
            save_video(gt_path, gt, args.enc_fps)
            save_video(pred_path, pred, args.dec_fps)

    deepspeed.comm.barrier()
    logger.info("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('-i', '--csv_path', type=str, required=True)
    parser.add_argument('-o', '--results_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/vfrtok-l.yaml')
    parser.add_argument('--ckpt', type=str, default='vfrtok-l.bin')
    parser.add_argument('--enc_fps', type=int, default=24)
    parser.add_argument('--dec_fps', type=int, default=24)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)