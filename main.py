# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Github source: https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from torch.utils.data import WeightedRandomSampler

from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from timematch_utils import label_utils
from timematch_utils.train_utils import bool_flag
from timematch_dataset import PixelSetData, create_evaluation_loaders
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    AddPixelLabels
)
from torchvision import transforms
from collections import Counter
from torch.utils import data
import random

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from unireplknet import *

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_data_loaders(splits, config, balance_source=True):

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
            AddPixelLabels()
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')

    return source_dataset , source_loader

def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    # 以下都是timematch
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=4096, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory')
    # parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
    #                     help='Path to datasets root directory')
    parser.add_argument('--source', default='france/30TXT/2017', type=str)
    parser.add_argument('--target', default='france/30TXT/2017', type=str)
    # 类别处理
    parser.add_argument('--combine_spring_and_winter', action='store_true')
    # 数据划分
    parser.add_argument('--num_folds', default=3, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    # 评估
    parser.add_argument('--sample_pixels_val', action='store_true')  # 布尔型开关参数（flag），它不需要传值，只需在命令行中出现或不出现该选项


    return parser


def main(args):
    utils.init_distributed_mode(args)

    # 只在主进程打印一次参数
    if utils.is_main_process():
        print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 注意：这里种子是全局固定的。如果你希望每个 Fold 的数据划分随机性不同，
    # 可以在 create_train_val_test_folds 内部处理随机种子，或者在这里根据 fold_num 动态调整。
    # 但通常为了复现性，我们固定一个全局种子，依靠 shuffle 在 folds 生成时打乱。
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    # 初始化日志 (Tensorboard/WandB)
    # 注意：如果是多折，WandB 可能需要为每个 fold 开启一个新的 run，或者在同一个 run 里区分 step
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        # 建议：在多折场景下，WandB 最好在循环内每次 re-init，或者手动管理 step
        # 这里暂时保持原样，但在循环内需要注意 step 重置或区分
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # ---------------------------------------------------------
    # 数据准备与 Fold 划分
    # ---------------------------------------------------------
    cfg = args
    config = args
    source_classes = label_utils.get_classes(cfg.source.split('/')[0],
                                             combine_spring_and_winter=cfg.combine_spring_and_winter)
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes)

    # 过滤类别
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)

    # 生成 K-Fold 划分
    indices = {config.source: len(source_data)}
    folds = create_train_val_test_folds([config.source], config.num_folds, indices, config.val_ratio, config.test_ratio)

    # 【重要】用于存储每个 Fold 的最佳准确率
    fold_best_accuracies = []

    # 【修正】计时开始：放在 Fold 循环之外，统计总耗时
    total_start_time = time.time()

    # =========================================================
    # 开始 K-Fold 循环
    # =========================================================
    for fold_num, splits in enumerate(folds):
        print(f'\n{"=" * 50}')
        print(f'Starting Fold {fold_num + 1} / {config.num_folds}')
        print(f'{"=" * 50}\n')

        config.fold_num = fold_num

        # 1. 为当前 Fold 创建独立的输出目录 (防止覆盖)
        if args.output_dir:
            fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_num}")
            os.makedirs(fold_output_dir, exist_ok=True)
            # 临时覆盖 args.output_dir 用于当前 fold 的保存
            original_output_dir = args.output_dir
            args.output_dir = fold_output_dir
        else:
            original_output_dir = None

        sample_pixels_val = config.sample_pixels_val

        # 2. 加载当前 Fold 的数据 (Dataset size 会变)
        val_dataset, val_loader, test_dataset, test_loader = create_evaluation_loaders(config.source, splits, config,
                                                                                       sample_pixels_val)
        source_dataset, source_loader = get_data_loaders(splits, config, config.balance_source)

        # 3. Mixup 配置
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=cfg.num_classes)  # 使用动态的 num_classes

        # 4. 创建模型 (每次都是新的随机初始化)
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=cfg.num_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
        )

        # 5. 加载预训练权重 (每次循环都重新加载，确保起点一致)
        if args.finetune:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                # 优化：如果文件很大，可以在循环外加载一次到内存，这里只做 copy
                checkpoint = torch.load(args.finetune, map_location='cpu')

            if utils.is_main_process():
                print("Load ckpt from %s" % args.finetune)

            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    if utils.is_main_process():
                        print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        model.to(device)

        # 6. EMA, DDP, Optimizer, Scheduler (全部基于当前 Fold 的数据量重建)
        model_ema = None
        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if utils.is_main_process():
            print("Model = %s" % str(model_without_ddp))
            print('number of params:', n_parameters)

        # 【关键】重新计算 Steps，因为 len(source_dataset) 每个 Fold 可能不同
        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(source_dataset) // total_batch_size

        if utils.is_main_process():
            print(
                f"[Fold {fold_num}] Dataset size: {len(source_dataset)}, Steps per epoch: {num_training_steps_per_epoch}")

        if args.layer_decay < 1.0 or args.layer_decay > 1.0:
            num_layers = 12
            assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
                "Layer Decay impl only supports convnext_small/base/large/xlarge"
            assigner = LayerDecayValueAssigner(
                list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              find_unused_parameters=False)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=None,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)

        loss_scaler = NativeScaler()

        # 【关键】为当前 Fold 生成独立的 LR 和 WD 曲线
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # 恢复 output_dir (如果之前修改过)，以便 auto_load_model 保存到正确的地方
        # 注意：auto_load_model 内部会使用 args.output_dir
        # 我们在上面已经修改了 args.output_dir = fold_output_dir，所以这里不需要改回去，直到 fold 结束

        # 自动加载断点 (如果有)
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

        max_accuracy = 0.0
        if args.model_ema and args.model_ema_eval:
            max_accuracy_ema = 0.0

        if utils.is_main_process():
            print("Start training for %d epochs" % args.epochs)

        fold_start_time = time.time()

        # 7. 训练循环
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                source_loader.sampler.set_epoch(epoch)

            # 更新 Log Writer 的 step (注意：如果是多折，step 可能会混淆，最好 reset 或者加 offset)
            if log_writer is not None:
                # 简单处理：每个 fold 重新开始计数，或者累加。这里选择累加以区分
                global_step_offset = fold_num * args.epochs * num_training_steps_per_epoch
                log_writer.set_step(global_step_offset + epoch * num_training_steps_per_epoch * args.update_freq)

            if wandb_logger:
                # WandB 可能需要手动处理 step，或者每个 fold 重新 init
                pass

            train_stats = train_one_epoch(
                model, criterion, source_loader, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, wandb_logger=wandb_logger,
                start_steps=(fold_num * args.epochs + epoch) * num_training_steps_per_epoch,  # 区分 fold 的 step
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                use_amp=args.use_amp
            )

            if args.output_dir and args.save_ckpt:
                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            if val_loader is not None:
                test_stats = evaluate(val_loader, model, device, use_amp=args.use_amp)
                if utils.is_main_process():
                    print(f"[Fold {fold_num}] Epoch {epoch}: Val Acc@1 {test_stats['acc1']:.1f}%")

                if max_accuracy < test_stats["acc1"]:
                    max_accuracy = test_stats["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

                if log_writer is not None:
                    log_writer.update(test_acc1=test_stats['acc1'], head=f"perf_fold_{fold_num}", step=epoch)

                # EMA eval logic ... (省略以保持简洁，逻辑同上)

            # 记录日志
            if args.output_dir and utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch, 'fold': fold_num, 'n_parameters': n_parameters}

                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if wandb_logger:
                    wandb_logger.log_epoch_metrics(log_stats)

        # ---------------------------------------------------------
        # 当前 Fold 结束，记录结果
        # ---------------------------------------------------------
        fold_best_accuracies.append(max_accuracy)
        fold_time = time.time() - fold_start_time
        if utils.is_main_process():
            print(
                f'>>> Fold {fold_num + 1} Finished. Best Acc: {max_accuracy:.2f}%. Time: {str(datetime.timedelta(seconds=int(fold_time)))}')

            # 恢复 output_dir 以便下一次循环创建新的子文件夹
            if original_output_dir:
                args.output_dir = original_output_dir

    # =========================================================
    # 所有 Fold 结束，计算统计指标
    # =========================================================
    if utils.is_main_process():
        print('\n' + '=' * 60)
        print('ALL FOLDS COMPLETED')
        print('=' * 60)
        print(f'Individual Fold Accuracies: {[f"{x:.2f}" for x in fold_best_accuracies]}')

        mean_acc = np.mean(fold_best_accuracies)
        std_acc = np.std(fold_best_accuracies)

        print(f'Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%')
        print('=' * 60)

        # 将最终结果写入一个总文件
        final_result_path = os.path.join(original_output_dir,
                                         "final_cv_results.txt") if original_output_dir else "final_cv_results.txt"
        with open(final_result_path, "w") as f:
            f.write(f"Folds: {fold_best_accuracies}\n")
            f.write(f"Mean: {mean_acc:.4f}\n")
            f.write(f"Std: {std_acc:.4f}\n")

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    total_time = time.time() - total_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Training Time for all folds: {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)