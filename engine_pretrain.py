import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (imgs, Keyword_list, Desc_list, Modality_list) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (imgs, Keyword_list, Desc_list) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        imgs = imgs.to(device, non_blocking=True).half()

        with torch.amp.autocast('cuda', ):
            clip_loss, clip_score = model(imgs, Keyword_list, Desc_list) #, Modality_list)
        loss =  clip_loss
        loss_value = loss.item()
        clip_loss_value = clip_loss.item()
        clip_score_value = clip_score.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad = 10, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(cliploss=clip_loss_value)
        metric_logger.update(clipscore=clip_score_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        clip_loss_value_reduce = misc.all_reduce_mean(clip_loss_value)
        clip_score_value_reduce = misc.all_reduce_mean(clip_score_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('clip_train_loss', clip_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('clip_train_score', clip_score_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # for data_iter_step, (imgs, Keyword_list, Desc_list, Modality_list) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (imgs, Keyword_list, Desc_list) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            clip_loss, clip_score = model(imgs, Keyword_list, Desc_list) #, Modality_list)
        loss = clip_loss
        loss_value = loss.item()
        clip_loss_value = clip_loss.item()
        clip_score_value = clip_score.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        torch.cuda.synchronize()
        metric_logger.update(cliploss=clip_loss_value)
        metric_logger.update(clipscore=clip_score_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        clip_loss_value_reduce = misc.all_reduce_mean(clip_loss_value)
        clip_score_value_reduce = misc.all_reduce_mean(clip_score_value)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('clip_val_loss', clip_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('clip_val_score', clip_score_value_reduce, epoch_1000x)
            # log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

