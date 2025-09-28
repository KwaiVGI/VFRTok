import deepspeed
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
#import wandb
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm 
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from einops import rearrange

import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import json
import math

from timm.scheduler import create_scheduler_v2 as create_scheduler

from data.dataset import ImageDataset, VideoDataset
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.data import random_crop_arr, center_crop_arr
from utils.runtime import build_logger, build_loss, build_model, get_deepspeed_latest_ckpt
from utils.runtime import log_gen_loss, log_disc_loss
from modelling.tokenizer import Tokenizers
import random

import warnings
warnings.filterwarnings('ignore')

def transcode(tensor: torch.Tensor,
               src_fps: float,
               dst_fps: float,
               t_patch_size: int) -> torch.Tensor:
    F = tensor.shape[2]
    assert 0 < dst_fps <= src_fps

    ratio = src_fps / dst_fps
    new_F = math.floor(F / ratio / t_patch_size) * t_patch_size
    new_fps = new_F / F * src_fps

    idx_float = torch.arange(new_F, dtype=torch.float32) * ratio
    idx = torch.floor(idx_float).long()
    idx = torch.clamp(idx, max=F-1)

    out = tensor[:, :, idx, :, :]
    return out, new_fps, idx

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP
    deepspeed.init_distributed()
    rank = deepspeed.comm.get_rank()

    with open(args.ds_config) as f:
        ds_config = json.load(f)

    # Setup an experiment folder:
    logger, tb_logger, checkpoint_dir = build_logger(args, rank)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting, world_size={deepspeed.comm.get_world_size()}.")

    # create and load model
    model = build_model(args)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f'Model: #Patch X -> #Token {model.encoder.num_latent_tokens}')
    if args.enc_token_drop > 0:
        logger.info(f'Enable token drop: [{args.enc_token_drop}, {args.enc_token_drop_max}]')

    loss_fn = build_loss(args)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in loss_fn.discriminator.parameters() if p.requires_grad):,}")

    optim_steps = 0
    if args.resume_exp: # get latest checkpoint
        ckpt_info = get_deepspeed_latest_ckpt(checkpoint_dir)
        optim_steps = ckpt_info['step']
    else:
        if args.model_ckpt: # load module
            checkpoint = torch.load(args.model_ckpt, map_location="cpu")
            model.load_state_dict(load_model_state_dict(checkpoint, model), strict=False)
            del checkpoint
            logger.info(f"Load checkpoint: {args.model_ckpt}")

        if args.disc_ckpt: # load disc
            checkpoint = torch.load(args.disc_ckpt, map_location="cpu")
            loss_fn.discriminator.load_state_dict(load_model_state_dict(checkpoint, None), strict=False)
            del checkpoint
            logger.info(f"Load discriminator checkpoint: {args.disc_ckpt}")

    # Setup data:
    if args.val_fps_list is None:
        args.val_fps_list = args.fps_list
    assert len(args.data_path_list) == len(args.fps_list) and len(args.val_data_path_list) == len(args.val_fps_list)
    if args.resume_exp: # random seed, avoid same data order
        sampler_seed = (args.global_seed + optim_steps) % 10000
    else:
        sampler_seed = args.global_seed

    train_loaders, train_iters, train_samplers, train_epochs = {}, {}, {}, {}
    loader_weight = {}
    for data_path, fps in zip(args.data_path_list, args.fps_list):
        num_frames = int(round(args.period * fps / args.t_patch_size)) * args.t_patch_size
        dataset = VideoDataset(data_path, args.data_column, args.image_size, num_frames, fps=fps, is_train=True, use_gpu=False)
        sampler = DistributedSampler(
            dataset,
            num_replicas=deepspeed.comm.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=sampler_seed
        )
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_size=int(ds_config['train_micro_batch_size_per_gpu']),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Dataset contains {len(dataset):,} records ({data_path})")
        logger.info(f"Set num_frames={num_frames} for fps={fps} and period={args.period}")
        train_loaders[fps] = loader
        train_iters[fps] = iter(loader)
        train_samplers[fps] = sampler
        train_epochs[fps] = 0
        loader_weight[fps] = len(loader)

    fps_array = np.array(args.fps_list)
    if args.fps_weight is not None:
        raw_weights = np.array(args.fps_weight)
    else:
        raw_weights = np.array([loader_weight[f] for f in fps_array])
    fps_prob = raw_weights / raw_weights.sum()
    logger.info(f"FPS Probs: {fps_prob.tolist()} ({fps_array.tolist()})")
    logger.info(f"Mix-FPS Rate: {args.mix_fps_rate}")
    logger.info(f"Symm Mix-FPS: {args.mix_symm_fps}")

    val_loaders = {}
    for data_path, fps in zip(args.val_data_path_list, args.val_fps_list):
        num_frames = int(round(args.period * fps / args.t_patch_size)) * args.t_patch_size
        val_dataset = VideoDataset(data_path, args.data_column, args.image_size, num_frames, fps=fps, is_train=False, use_gpu=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        logger.info(f"Val Dataset contains {len(val_dataset):,} records ({data_path})")
        logger.info(f"Set num_frames={num_frames} for fps={fps} and period={args.period}")
        val_loaders[fps] = val_loader

    # 初始化deepspeed engine
    grad_acc_steps = ds_config['gradient_accumulation_steps']
    max_train_steps = args.max_train_steps #len(loader) * args.epochs
    max_optim_steps = max_train_steps // grad_acc_steps
    ds_config['scheduler']['params']['total_num_steps'] = max_optim_steps
    ds_config['train_batch_size'] = \
        ds_config['train_micro_batch_size_per_gpu'] * grad_acc_steps * deepspeed.comm.get_world_size()
    logger.info(f"Train batch size: {ds_config['train_micro_batch_size_per_gpu']}x{grad_acc_steps}x{deepspeed.comm.get_world_size()}={ds_config['train_batch_size']}.")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )
    logger.info(f"Finish model deepspeed initialization.")
    disc_engine, disc_optimizer, _, _ = deepspeed.initialize(
        model=loss_fn,
        model_parameters=loss_fn.discriminator.parameters(),
        config_params=ds_config
    )
    logger.info(f"Finish Discriminator deepspeed initialization.")
    dtype = next(model_engine.module.parameters()).dtype
    device = model_engine.device

    # resume from deepspeed
    if args.resume_exp:
        model_engine.load_checkpoint(checkpoint_dir, ckpt_info['model_ckpt'], load_module_strict=False)
        disc_engine.load_checkpoint(checkpoint_dir, ckpt_info['disc_ckpt'])
        logger.info(f"Resume training from {args.resume_exp}")

    # set step
    # start_epoch = math.ceil(optim_steps / len(loader))
    start_step = optim_steps % max_optim_steps
    train_steps = optim_steps * grad_acc_steps
    logger.info(f"Initial state: steps={optim_steps}")
    
    # Variables for monitoring/logging purposes:
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {max_train_steps} steps...")
    while optim_steps < max_train_steps:
        # fps = random.choice(args.fps_list)
        fps_idx = np.random.choice(len(fps_array), p=fps_prob)
        fps, prob = fps_array[fps_idx], fps_prob[fps_idx]
        try:
            batch = next(train_iters[fps])
        except StopIteration:
            train_epochs[fps] += 1
            train_samplers[fps].set_epoch(train_epochs[fps])
            train_iters[fps] = iter(train_loaders[fps])
            batch = next(train_iters[fps])

        if args.mix_fps:
            '''
            混合FIX训练:Encode和Decode采用的fps不一定相同
            '''
            if prob >= 0.2 and random.random() <= args.mix_fps_rate: # force 24 fps
                batch, fps, _ = transcode(batch, fps, 24, args.t_patch_size)
                enc_batch = dec_batch = batch
                enc_fps = dec_fps = fps
                enc_num_frames = dec_num_frames = batch.shape[2]
            else:
                fps_candidates = [x for x in range(12, fps+1, 6)] # this works when args.t_patch_size=4
                low_fps = random.choice(fps_candidates)
                low_fps_batch, low_fps, low_fps_frame_pts = transcode(batch, fps, low_fps, args.t_patch_size)
                if args.mix_symm_fps:
                    enc_batch = dec_batch = low_fps_batch
                    enc_fps = dec_fps = low_fps
                    enc_num_frames = dec_num_frames = enc_batch.shape[2]
                elif random.random() > 0.5: # low -> high
                    enc_batch, dec_batch = low_fps_batch, batch
                    enc_fps, dec_fps = low_fps, fps
                    enc_num_frames, dec_num_frames = enc_batch.shape[2], batch.shape[2]
                else: # high -> low
                    enc_batch, dec_batch = batch, low_fps_batch
                    enc_fps, dec_fps = fps, low_fps
                    enc_num_frames, dec_num_frames = batch.shape[2], dec_batch.shape[2]
        else:
            enc_batch = dec_batch = batch
            enc_fps = dec_fps = fps
            enc_num_frames = dec_num_frames = batch.shape[2]

        train_steps += 1
        optim_steps = train_steps // grad_acc_steps

        imgs = enc_batch.to(device=device, dtype=dtype)
        gts = dec_batch.to(device=device, dtype=dtype)
        recons_imgs, codebook_loss, _ = model_engine(imgs, enc_num_frames, enc_fps, dec_num_frames, dec_fps)

        loss_gen_info = disc_engine(codebook_loss, gts, recons_imgs, optimizer_idx=0, global_step=optim_steps, 
                                last_layer=model_engine.module.decoder.last_layer)
        loss_gen = loss_gen_info['loss']
        model_engine.backward(loss_gen)

        disc_optimizer.zero_grad() # remove the grad of discriminator
        loss_disc_info = disc_engine(codebook_loss, gts, recons_imgs, optimizer_idx=1, global_step=optim_steps)
        loss_disc = loss_disc_info['discriminator_adv_loss']
        disc_engine.backward(loss_disc)

        model_engine.step()
        model_engine.zero_grad()
        disc_engine.step()
        disc_engine.zero_grad()

        # Log loss values:
        running_loss += loss_gen.item() + loss_disc.item()
            
        if train_steps % grad_acc_steps == 0 and optim_steps % args.log_every == 0:
            if rank == 0:
                log_gen_loss(loss_gen_info, optim_steps, logger, tb_logger)
                log_disc_loss(loss_disc_info, optim_steps, logger, tb_logger)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = args.log_every / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / args.log_every, device=model_engine.device)
            deepspeed.comm.all_reduce(avg_loss, op=deepspeed.comm.ReduceOp.SUM)
            avg_loss = avg_loss.item() / deepspeed.comm.get_world_size()
            logger.info(f"(step={optim_steps:07d}/total_steps:{max_optim_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            start_time = time.time()
        
            if rank == 0 and tb_logger is not None:
                tb_logger.add_scalar("LR", optimizer.param_groups[0]["lr"], optim_steps)
                tb_logger.add_scalar("Train/Loss", avg_loss, optim_steps)
                
        if train_steps % grad_acc_steps == 0 and optim_steps % args.val_every == 0:
            model_engine.eval()
            for val_fps in args.val_fps_list:
                val_ins, val_outs = [], []
                val_loader = val_loaders[val_fps]
                val_num_frames = None
                for val_in in tqdm(val_loader, total=len(val_loader), desc='Val'):
                    val_num_frames = val_in.shape[2] # b c f h w
                    val_in = val_in.to(device=device, dtype=dtype)
                    with torch.no_grad():
                        val_out, _, _ = model_engine(val_in, val_num_frames, val_fps, val_num_frames, val_fps)
                    val_ins.append(val_in)
                    val_outs.append(val_out)
                val_ins = torch.cat(val_ins, dim=0)
                val_outs = torch.cat(val_outs, dim=0)
                val_ins = (torch.clamp(val_ins, min=-1, max=1) + 1) / 2
                val_outs = (torch.clamp(val_outs, min=-1, max=1) + 1) / 2
                val_ins, val_outs = val_ins.cpu().float(), val_outs.cpu().float()
                # visualize
                if val_num_frames >= 4: # b c f h w
                    step = val_num_frames // 4
                    vis_img = torch.stack([
                        val_ins[0, :, ::step], val_ins[1, :, ::step], val_outs[0, :, ::step], val_outs[1, :, ::step],
                        val_ins[2, :, ::step], val_ins[3, :, ::step], val_outs[2, :, ::step], val_outs[3, :, ::step],
                        val_ins[4, :, ::step], val_ins[5, :, ::step], val_outs[4, :, ::step], val_outs[5, :, ::step],
                        val_ins[6, :, ::step], val_ins[7, :, ::step], val_outs[6, :, ::step], val_outs[7, :, ::step],
                    ], dim=0)
                    vis_img = rearrange(vis_img, 'b c f h w -> (b f) c h w')
                    vis_img = make_grid(vis_img, nrow=8, padding=0, pad_value=1.0) # 8x8
                elif val_num_frames > 1: # b c f h w
                    vis_img = torch.stack([
                        val_ins[0, :, :2], val_ins[1, :, :2], val_outs[0, :, :2], val_outs[1, :, :2],
                        val_ins[2, :, :2], val_ins[3, :, :2], val_outs[2, :, :2], val_outs[3, :, :2],
                        val_ins[4, :, :2], val_ins[5, :, :2], val_outs[4, :, :2], val_outs[5, :, :2],
                        val_ins[6, :, :2], val_ins[7, :, :2], val_outs[6, :, :2], val_outs[7, :, :2],
                    ], dim=0)
                    vis_img = rearrange(vis_img, 'b c f h w -> (b f) c h w')
                    vis_img = make_grid(vis_img, nrow=4, padding=0, pad_value=1.0) # 8x8
                else:
                    vis_img = torch.cat([
                        val_ins[:4], val_outs[:4], val_ins[4:8], val_outs[4:8]
                    ], dim=0)
                    vis_img = make_grid(vis_img, nrow=4, padding=0, pad_value=1.0) # 4x4
                # vis_img = vis_img.permute(1, 2, 0).mul_(255).numpy().astype(np.uint8)
                vis_img = vis_img.mul_(255).to(torch.uint8)
                if rank == 0:
                    # wandb_logger.log({"recon_images": [wandb.Image(image)]}, step=optim_steps)
                    # vis_img = torch.from_numpy(vis_img).permute(2, 0, 1)  # 转换为 CHW
                    tb_logger.add_image(f"Recon_Images@{val_fps:.2f}", vis_img, optim_steps, dataformats='CHW')

                # calculate metrics
                val_psnr, val_ssim = [], []
                val_ins = rearrange(val_ins, 'b c f h w -> (b f) h w c')
                val_outs = rearrange(val_outs, 'b c f h w -> (b f) h w c')
                val_ins, val_outs = val_ins.numpy(), val_outs.numpy()
                for val_in, val_out in zip(val_ins, val_outs):
                    psnr = psnr_loss(val_out, val_in)
                    ssim = ssim_loss(val_out, val_in, multichannel=True, data_range=1.0, channel_axis=-1) # FIXME: experiments<=023, data_range was 2.0
                    val_psnr.append(psnr)
                    val_ssim.append(ssim)

                if rank == 0:
                    val_psnr, val_ssim = np.mean(val_psnr), np.mean(val_ssim)
                    tb_logger.add_scalar(f"Val/PSNR@{val_fps:.2f}", val_psnr, optim_steps)
                    tb_logger.add_scalar(f"Val/SSIM@{val_fps:.2f}", val_ssim, optim_steps)
                    logger.info(f"[Validation] PSNR@{val_fps:.2f}: {val_psnr:.2f}, SSIM@{val_fps:.2f}: {val_ssim:.4f}")
            model_engine.train()

        # Save checkpoint:
        if train_steps % grad_acc_steps == 0 and optim_steps % args.ckpt_every == 0 and train_steps > 0:
            model_engine.save_checkpoint(checkpoint_dir, f"model_{optim_steps:07d}")
            disc_engine.save_checkpoint(checkpoint_dir, f"disc_{optim_steps:07d}")
            deepspeed.comm.barrier()

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    parser.add_argument("--ds-config", type=str, default='/configs/deepspeed/ds_config.json')
    
    parser.add_argument("--resume-exp", type=str, default=None, help="resume experiment name")
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    # parser.add_argument("--data-path", type=str, default="/ytech_m2v2_hdd/dataset/imagenet-1k/train.csv")
    parser.add_argument("--data-path-list", type=str, nargs='+', default=[])
    parser.add_argument("--fps-list", type=float, nargs='+', default=[])
    parser.add_argument("--fps-weight", type=float, nargs='+', default=None)
    # parser.add_argument("--val-data-path", type=str, default="/ytech_m2v2_hdd/zhongtianxiong/VAE/continuous_tokenizer/data/vpg_val_psy_img.csv")
    parser.add_argument("--val-data-path-list", type=str, nargs='+', default=[])
    parser.add_argument("--val-fps-list", type=float, nargs='+', default=None)
    parser.add_argument("--data-column", type=str, default="image_path")
    parser.add_argument("--model", type=str, choices=list(Tokenizers.keys()), default="VQ-16")
    parser.add_argument("--model-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--disc-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", type=str2bool, default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--vq-loss-ratio", type=float, default=1.0, help="vq loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--kl-loss-weight", type=float, default=0.000001)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-codebooks", type=int, default=1)
    
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--perceptual-loss", type=str, default='vgg', help="perceptual loss type of LPIPS", choices=['vgg', 'timm', 'tv'])
    parser.add_argument("--perceptual-model", type=str, default='vgg', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-dino-variants", type=str, default='depth12_no_train', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-intermediate-loss", type=str2bool, default=False, help="perceptual loss compute at intermedia features of LPIPS")
    parser.add_argument("--perceptual-logit-loss", type=str2bool, default=False, help="perceptual loss compute at logits of LPIPS")
    parser.add_argument("--perceptual-resize", type=str2bool, default=False, help="perceptual loss compute at resized images of LPIPS")
    parser.add_argument("--perceptual-warmup", type=int, default=None, help="iteration to warmup perceptual loss")
    
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-dim", type=int, default=64, help="discriminator channel base dimension")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--lecam-loss-weight", type=float, default=None)
    parser.add_argument("--use-diff-aug",type=str2bool, default=False)
    parser.add_argument("--disc-cr-loss-weight", type=float, default=0.0, help="discriminator consistency loss weight for gan training")
    parser.add_argument("--disc-adaptive-weight",type=str2bool, default=False)

    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="experiments")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--period", type=float, default=1)
    # parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=500000)
    parser.add_argument("--variable-num-frames", type=str2bool, default=False)
    parser.add_argument("--mix-fps", type=str2bool, default=False)
    parser.add_argument("--mix-fps-rate", type=float, default=1)
    parser.add_argument("--mix-symm-fps", type=str2bool, default=False)
    parser.add_argument("--use-coord-mlp", type=str2bool, default=False)
    
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--t-patch-size", type=int, default=1, help='temporal patch size')

    parser.add_argument("--dec-seperate-mask-token", type=str2bool, default=False)

    # Normal dist.
    parser.add_argument("--enc-token-drop", type=float, default=0.0, help='ratio of tokens to mask')
    parser.add_argument("--enc-token-drop-max", type=float, default=0.6, help='max ratio of tokens to mask')

    # Uniform dist.
    parser.add_argument("--latent-token-drop-max", type=float, default=0.0, help='max ratio of latents tokens to mask')

    # position embedding
    parser.add_argument("--rope-mixed", type=str2bool, default=False)
    parser.add_argument("--rope-dim", type=int, default=None)
    parser.add_argument("--rope-heads", type=int, default=None)
    parser.add_argument("--rope-layers", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10.0)
    parser.add_argument("--rope-theta-t", type=float, default=100.0)
    parser.add_argument("--to-pixel", type=str, default='linear')
    parser.add_argument("--use-ape", type=str2bool, default=False)
    parser.add_argument("--use-rope", type=str2bool, default=True)

    #First parse of command-line args to check for config file
    args = parser.parse_args()

    if args.resume_exp:
        args.config = os.path.join(args.resume_exp, 'config.yaml')
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
