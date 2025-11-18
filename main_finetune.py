import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import FinetuneMMFundusDataset
from models.downstream import DownstreamModule
import random
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from engine_finetune import train_one_epoch, val_one_epoch, test_epoch
from util.adopt import ADOPT
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import warnings


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('imagebind-llm pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--visual_encoder', default='clip_vit', type=str,
                        help='Name of visual encoder to train')
    parser.add_argument('--text_encoder', default='clip_text', type=str,
                        help='Name of text encoder to train')
    parser.add_argument('--is_probe', default=False, type=str2bool,
                        help='whether to use linear probe')
    parser.add_argument('--default_pretrain', default=None, type=str, required=False,
                        help='whether to use default pretrain')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--test_only', default=False, type=str2bool,
                        help='whether to test only')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')
    # Optimizer parameters
    parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Name of optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_dir', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--csv_name', default=None, type=str,
                        help='csv file name')
    parser.add_argument('--task', default='multilabel', type=str, choices=['multilabel', 'multiclass'],
                        help='task type')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use_checkpoint', default=False, type=bool)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define the model
    model = DownstreamModule(visual_encoder=args.visual_encoder, task=args.task, num_classes=args.num_classes, is_probe=args.is_probe, default_pretrain=args.default_pretrain)
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        message = model.load_state_dict(checkpoint["model"], strict=False)
        print(message)
    
    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optimizer == 'adopt':
        optimizer = ADOPT(param_groups, lr=args.lr, betas=(0.9, 0.95), decouple=True)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    print(optimizer)
    loss_scaler = NativeScaler()
    
    
    # Resume from a checkpoint
    start_epoch = args.start_epoch
    if args.resume:
        start_epoch, model, optimizer, loss_scaler = misc.load_model(
            args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    for param_id, state in optimizer.state.items():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    if args.test_only:
        num_epoch = start_epoch
    else:
        num_epoch = args.epochs
    
    model.set_default_trainability()
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.get_trainable_params().items()])
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # create data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-30, 30)),
        # transforms.GaussianBlur(kernel_size=5),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                     antialias=None),  # 3 is bicubic
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    transform_val = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=BICUBIC),  # 3 is bicubic
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_train = FinetuneMMFundusDataset(data_dir=args.data_dir, csv_name=args.csv_name, transform=transform_train, partition='train')
    dataset_val = FinetuneMMFundusDataset(data_dir=args.data_dir, csv_name=args.csv_name, transform=transform_val, partition='val')
    dataset_test = FinetuneMMFundusDataset(data_dir=args.data_dir, csv_name=args.csv_name, transform=transform_val, partition='test')
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    sample_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

    print("Sampler_train = %s" % str(sampler_train))


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sample_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # SummaryWrite
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None


    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    print(f"Start training for {num_epoch} epochs")
    start_time = time.time()
    best_acc = 0
    
    for epoch in range(start_epoch, num_epoch):
        device = torch.device(f"cuda:{misc.get_rank()}")
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        val_stats = val_one_epoch(
            model, data_loader_val,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and ((epoch+1) % 10 == 0):

            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        
        if val_stats['epoch_acc'] > best_acc:
            best_acc = val_stats['epoch_acc']
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, is_best=True)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': round(v, 6) if isinstance(v, float) else v for k, v in train_stats.items()},
                     **{f'val_{k}': round(v, 6) if isinstance(v, float) else v for k, v in val_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    model = misc.load_best_model(model.module, args.output_dir)
    test_stats = test_epoch(
        model, data_loader_test,
        device, log_writer=log_writer,
        args=args
    )
    log_stats = {**{f'test_{k}': round(v, 6) if isinstance(v, float) else v for k, v in test_stats.items()}}
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")
    warnings.filterwarnings("ignore", message="Only one class is present in y_true. ROC AUC score is not defined in that case.")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if not args.test_only:
        json_path = os.path.join(args.output_dir, 'args.json')
        with open(json_path,'w') as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4, separators=(',', ':')))
    main(args)
