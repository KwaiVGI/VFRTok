from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml as yaml

import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import deepspeed
import torch.distributed as dist

from utils.logger_func import create_logger
from modelling.tokenizer import Tokenizers
from losses.loss import Loss

def log_gen_loss(loss_info, global_step, logger, tb_logger):
    log_str = '(Generator)'
    for name, loss in loss_info.items():
        if name == 'loss':
            continue
        log_str += f' {name}: {loss:.4f}'
        if tb_logger is not None:
            tb_logger.add_scalar(name, loss, global_step)
    logger.info(log_str)

def log_disc_loss(loss_info, global_step, logger, tb_logger):
    log_str = '(Discriminator)'
    for name, loss in loss_info.items():
        log_str += f' {name}: {loss:.4f}'
    logger.info(log_str)
    if tb_logger is not None:
        tb_logger.add_scalar('discriminator_adv_loss', loss_info['discriminator_adv_loss'], global_step)

def build_logger(args, rank):
    if args.resume_exp:
        experiment_dir = args.resume_exp
    else:
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            experiment_index = -1
            for exist_exp in os.listdir(args.results_dir):
                if 'exp' in exist_exp:
                    exist_index = int(exist_exp.split('-')[0].replace('exp', ''))
                    experiment_index = max(experiment_index, exist_index)
            experiment_index += 1
        dist.barrier() # make sure the same index
        if args.config is not None:
            model_string_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_string_name.startswith('exp'):
                model_string_name = '-'.join(model_string_name.split('-')[1:])
        else:
            model_string_name = args.model.replace("/", "-")
        # Create an experiment folder
        if args.ds_config:
            ds_string_name = '.'.join(args.ds_config.split('/')[-1].split('.')[:-1])
            experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}-{ds_string_name}"
        else:
            experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        
        #wandb_logger = wandb.init(project='tokenizer', name=f'exp{experiment_index:03d}-{model_string_name}')
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
        tb_logger = SummaryWriter(log_dir=os.path.join(experiment_dir, "logs"))
    else:
        logger, tb_logger = create_logger(None), None
    return logger, tb_logger, checkpoint_dir

def build_model(args):
    return Tokenizers[args.model](
        image_size=args.image_size,
        num_frames=args.num_frames,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        t_patch_size=args.t_patch_size,
        tau=args.tau,
        num_codebooks=args.num_codebooks,
        use_ape=args.use_ape,
        use_rope=args.use_rope,
        rope_mixed=args.rope_mixed,
        rope_dim=args.rope_dim,
        rope_heads=args.rope_heads,
        rope_layers=args.rope_layers,
        rope_theta=args.rope_theta,
        rope_theta_t=args.rope_theta_t,
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        latent_token_drop_max=args.latent_token_drop_max,
        dec_seperate_mask_token=args.dec_seperate_mask_token,
        variable_num_frames=args.variable_num_frames if hasattr(args, 'variable_num_frames') else False,
        use_coord_mlp=args.use_coord_mlp if hasattr(args, 'use_coord_mlp') else False,
    )

def build_loss(args):
    return Loss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        disc_dim=args.disc_dim,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        perceptual_loss=args.perceptual_loss,
        perceptual_model=args.perceptual_model,
        perceptual_dino_variants=args.perceptual_dino_variants,
        perceptual_weight=args.perceptual_weight,
        perceptual_intermediate_loss=args.perceptual_intermediate_loss,
        perceptural_logit_loss=args.perceptual_logit_loss,
        perceptual_resize=args.perceptual_resize,
        perceptual_warmup=args.perceptual_warmup,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_cr_loss_weight=args.disc_cr_loss_weight,
        use_diff_aug=args.use_diff_aug,
        disc_adaptive_weight=args.disc_adaptive_weight
    )


def get_deepspeed_latest_ckpt(checkpoint_dir):
    ckpts = os.listdir(checkpoint_dir)
    model_ckpt = sorted([x for x in ckpts if 'model' in x])[-1]
    disc_ckpt = sorted([x for x in ckpts if 'disc' in x])[-1]
    model_step = int(model_ckpt.replace('model_', ''))
    disc_step = int(disc_ckpt.replace('disc_', ''))
    assert model_step == disc_step
    ckpt_info = {
        'type': 'deepspeed',
        'step': model_step,
        'model_ckpt': model_ckpt, 
        'disc_ckpt': disc_ckpt, 
    }
    return ckpt_info

def get_ckpt(checkpoint_dir, step=-1):
    ckpts = os.listdir(checkpoint_dir)
    if 'zero_to_fp32.py' in ckpts: # deepspeed
        if step == -1: # latest
            model_ckpt = sorted([x for x in ckpts if 'model' in x])[-1]
            disc_ckpt = sorted([x for x in ckpts if 'disc' in x])[-1]
            step = int(model_ckpt.replace('model_', ''))
        else:
            model_ckpt = f'model_{step:07d}'
            disc_ckpt = f'disc_{step:07d}'
        model_ckpt = os.path.join(checkpoint_dir, model_ckpt, 'pytorch_model.bin')
        disc_ckpt = os.path.join(checkpoint_dir, disc_ckpt, 'pytorch_model.bin')

        model_ckpt = torch.load(model_ckpt, map_location="cpu")
        disc_ckpt = torch.load(disc_ckpt, map_location="cpu")

        ckpt_info = {
            'type': 'deepspeed',
            'step': step,
            'model_ckpt': model_ckpt, 
            'disc_ckpt': disc_ckpt, 
        }

    else: # torchrun
        if step == -1: # latest
            ckpts = [x for x in ckpts if x.endswith('.pt')]
            ckpt = sorted(ckpts)[-1]
            step = int(ckpt.replace('.pt', ''))
        else:
            ckpt = f'{step:07d}.pt'
        ckpt = os.path.join(checkpoint_dir, ckpt)

        ckpt = torch.load(ckpt, map_location="cpu")
        model_ckpt = ckpt['model']
        disc_ckpt = ckpt['discriminator']
        opt_ckpt = ckpt['optimizer']
        opt_disc_ckpt = ckpt['optimizer_disc']

        ckpt_info = {
            'type': 'torchrun',
            'step': step,
            'model_ckpt': model_ckpt, 
            'disc_ckpt': disc_ckpt, 
            'opt_ckpt': opt_ckpt, 
            'opt_disc_ckpt': opt_disc_ckpt
        }
    return ckpt_info