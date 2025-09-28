import torch
import os
import numpy as np
import random

import datetime
import functools
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as tdist

import argparse
from functools import reduce



def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints = [f for f in checkpoints if 'best_ckpt' not in f]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


def get_valid_param(orig_state_dict, model, key):
    ckpt_param = orig_state_dict[key]
    module_param = reduce(getattr, key.split('.'), model)
    if ckpt_param.shape == module_param.shape:
        return ckpt_param
    else:
        # print(f'{key}: ckpt_shape ({ckpt_param.shape}) != module_shape ({module_param.shape})')
        return module_param

def load_model_state_dict(orig_state_dict, model=None):
    model_state = {}
    for key, value in orig_state_dict.items():
        if key.startswith("module."):
            model_state[key[7:]] = value
        if key.startswith("_orig_mod."):
            model_state[key[10:]] = value
        if key.startswith("discriminator."):
            model_state[key[14:]] = value
        elif model and key in [
            'encoder.model.pos_embed', 'decoder.model.pos_embed', 
            'encoder.latent_tokens', 'decoder.latent_tokens',
            'encoder.latent_pos_embed', 'decoder.latent_pos_embed',
            'encoder.model.patch_embed.proj.weight', 'decoder.model.patch_embed.proj.weight',
            'decoder.to_pixel.model.weight', 'decoder.to_pixel.model.bias',
            'decoder.mask_token',
            'quant_conv.weight', 'quant_conv.bias',
            'post_quant_conv.weight', 'post_quant_conv.bias',
        ]:
            model_state[key] = get_valid_param(orig_state_dict, model, key)
        elif model and hasattr(model, 'bit_estimator') and key in [
            'bit_estimator.f1.h', 'bit_estimator.f1.a', 'bit_estimator.f1.b',
            'bit_estimator.f2.h', 'bit_estimator.f2.a', 'bit_estimator.f2.b',
            'bit_estimator.f3.h', 'bit_estimator.f3.a', 'bit_estimator.f3.b',
            'bit_estimator.f4.h', 'bit_estimator.f4.b',
        ]:
            model_state[key] = get_valid_param(orig_state_dict, model, key)
        else:
            model_state[key] = value
    if model:
        if hasattr(model.encoder, 'mask_token'):
            model_state['encoder.mask_token'] = model.encoder.mask_token
        if hasattr(model, 'bit_estimator') and 'bit_estimator' not in model_state:
            for key in [
                'bit_estimator.f1.h', 'bit_estimator.f1.a', 'bit_estimator.f1.b',
                'bit_estimator.f2.h', 'bit_estimator.f2.a', 'bit_estimator.f2.b',
                'bit_estimator.f3.h', 'bit_estimator.f3.a', 'bit_estimator.f3.b',
                'bit_estimator.f4.h', 'bit_estimator.f4.b',
            ]:
                model_state[key] = reduce(getattr, key.split('.'), model)
        
    return model_state