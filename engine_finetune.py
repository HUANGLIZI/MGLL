import math
import sys
from typing import Iterable
import os
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import Accuracy


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
    accum_iter = args.accum_iter # type: ignore

    optimizer.zero_grad()
    if args.task == 'multiclass': # type: ignore # typ # type: ignoree: ignore
        accuracy_metric = Accuracy(task=args.task, num_classes=args.num_classes) # type: ignore
    elif args.task == 'multilabel': # type: ignore
        accuracy_metric = Accuracy(task=args.task, num_labels=args.num_classes, average='macro')
        # strict_accuracy
    else:
        raise ValueError(f"Invalid task: {args.task}") # type: ignore
    all_labels = []
    all_probs = []

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # print(device)
    for data_iter_step, (imgs, labels, name) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args) # type: ignore
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.task == 'multiclass':
            labels = labels[:, 0].long()

        with torch.amp.autocast('cuda', ):
            logits, ce_loss = model(imgs, labels)
        loss = ce_loss
        loss_value = loss.item()
        if args.task == 'multiclass':
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
            
        batch_acc = accuracy_metric(probs.detach().cpu(), labels.detach().cpu()).item()
        labels_np = labels.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        all_labels.append(labels_np)
        all_probs.append(probs_np)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad = 10, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(ce_loss=loss_value)
        metric_logger.update(acc=batch_acc)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        acc_value_reduce = misc.all_reduce_mean(batch_acc)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_ce_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', acc_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    all_labels_np = torch.tensor(np.concat(all_labels, axis=0)).to(device)
    all_probs_np = torch.tensor(np.concat(all_probs, axis=0)).to(device)
    gathered_labels = [torch.zeros_like(all_labels_np) for _ in range(torch.distributed.get_world_size())]
    gathered_probs = [torch.zeros_like(all_probs_np) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_labels, all_labels_np)
    torch.distributed.all_gather(gathered_probs, all_probs_np)
    gathered_labels_np = torch.cat(gathered_labels).cpu().numpy()
    gathered_probs_np = torch.cat(gathered_probs).cpu().numpy()

    try:
        if gathered_probs_np.shape[1] == 2 and args.task == 'multiclass':
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
        else:
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np, average="macro", multi_class="ovr")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np, average="macro")
    except ValueError:
        epoch_auc = float('nan')
        epoch_map = float('nan')

    global_probs = torch.tensor(gathered_probs_np)
    global_labels = torch.tensor(gathered_labels_np)
    epoch_acc = accuracy_metric(global_probs, global_labels).item()
    
    metric_logger.synchronize_between_processes()
    print("[Train] Averaged stats:", metric_logger)
    print("[Train] Global stats: epoch_auc={}, epoch_map={}, epoch_acc={}".format(epoch_auc, epoch_map, epoch_acc))
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict['epoch_auc'] = epoch_auc
    return_dict['epoch_map'] = epoch_map
    return_dict['epoch_acc'] = epoch_acc
    return return_dict


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
    if args.task == 'multiclass':
        accuracy_metric = Accuracy(task=args.task, num_classes=args.num_classes)
    elif args.task == 'multilabel':
        accuracy_metric = Accuracy(task=args.task, num_labels=args.num_classes, average='macro')
    else:
        raise ValueError(f"Invalid task: {args.task}")
    all_labels = []
    all_probs = []

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (imgs, labels, name) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.task == 'multiclass':
            labels = labels[:, 0].long()

        with torch.no_grad():
            logits, ce_loss = model(imgs, labels)
        loss = ce_loss
        loss_value = loss.item()
        if args.task == 'multiclass':
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
            
        batch_acc = accuracy_metric(probs.detach().cpu(), labels.detach().cpu()).item()
        labels_np = labels.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        all_labels.append(labels_np)
        all_probs.append(probs_np)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter
        torch.cuda.synchronize()

        metric_logger.update(ce_loss=loss_value)
        metric_logger.update(acc=batch_acc)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        acc_value_reduce = misc.all_reduce_mean(batch_acc)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_ce_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('val_acc', acc_value_reduce, epoch_1000x)

    # gather the stats from all processes
    all_labels_np = torch.tensor(np.concat(all_labels, axis=0)).to(device)
    all_probs_np = torch.tensor(np.concat(all_probs, axis=0)).to(device)
    gathered_labels = [torch.zeros_like(all_labels_np) for _ in range(torch.distributed.get_world_size())]
    gathered_probs = [torch.zeros_like(all_probs_np) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_labels, all_labels_np)
    torch.distributed.all_gather(gathered_probs, all_probs_np)
    gathered_labels_np = torch.cat(gathered_labels).cpu().numpy()
    gathered_probs_np = torch.cat(gathered_probs).cpu().numpy()

    try:
        if gathered_probs_np.shape[1] == 2 and args.task == 'multiclass':
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
        else:
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np, average="macro", multi_class="ovr")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np, average="macro")
    except ValueError:
        epoch_auc = float('nan')
        epoch_map = float('nan')

    global_labels = torch.tensor(gathered_labels_np)
    global_probs = torch.tensor(gathered_probs_np)
    epoch_acc = accuracy_metric(global_probs, global_labels).item()
    
    metric_logger.synchronize_between_processes()
    print("[Val] Averaged stats:", metric_logger)
    print("[Val] Global stats: epoch_auc={}, epoch_map={}, epoch_acc={}".format(epoch_auc, epoch_map, epoch_acc))
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict['epoch_auc'] = epoch_auc
    return_dict['epoch_map'] = epoch_map
    return_dict['epoch_acc'] = epoch_acc
    return return_dict


def test_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, 
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [Test]'
    print_freq = 20
    max_name_length = 30

    accum_iter = args.accum_iter
    if args.task == 'multiclass':
        accuracy_metric = Accuracy(task=args.task, num_classes=args.num_classes)
    elif args.task == 'multilabel':
        accuracy_metric = Accuracy(task=args.task, num_labels=args.num_classes, average='macro')
    else:
        raise ValueError(f"Invalid task: {args.task}")
    
    all_labels = []
    all_probs = []
    all_names = []

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (imgs, labels, name) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.task == 'multiclass':
            labels = labels[:, 0].long()

        with torch.no_grad():
            logits, ce_loss = model(imgs, labels)
        loss = ce_loss
        loss_value = loss.item()
        if args.task == 'multiclass':
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
        labels_np = labels.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        name_encoded = [n.encode('utf-8') for n in name]
        padded_names = [n[:max_name_length] if len(n) > max_name_length else n.ljust(max_name_length, b'\x00') for n in name_encoded]
        padded_names_tensor = torch.tensor([list(n) for n in padded_names], dtype=torch.uint8, device=device)
        all_labels.append(labels_np)
        all_probs.append(probs_np)
        all_names.append(padded_names_tensor)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping testing".format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter
        torch.cuda.synchronize()
        metric_logger.update(bce_loss=loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

    # Gather the stats from all processes
    all_labels_np = torch.tensor(np.concat(all_labels, axis=0)).to(device)
    all_probs_np = torch.tensor(np.concat(all_probs, axis=0)).to(device)
    all_names_np = torch.tensor(torch.concatenate(all_names)).reshape(-1, max_name_length).to(device)
    gathered_names = [torch.zeros_like(all_names_np) for _ in range(torch.distributed.get_world_size())]
    gathered_labels = [torch.zeros_like(all_labels_np) for _ in range(torch.distributed.get_world_size())]
    gathered_probs = [torch.zeros_like(all_probs_np) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_labels, all_labels_np)
    torch.distributed.all_gather(gathered_probs, all_probs_np)
    torch.distributed.all_gather(gathered_names, all_names_np)
    gathered_names_np = torch.cat(gathered_names).cpu().numpy()
    gathered_labels_np = torch.cat(gathered_labels).cpu().numpy()
    gathered_probs_np = torch.cat(gathered_probs).cpu().numpy()
    decoded_names = [bytes(name).decode('utf-8').rstrip('\x00') for name in gathered_names_np]

    # Compute global metrics
    try:
        if gathered_probs_np.shape[1] == 2 and args.task == 'multiclass':
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np[:, 1], average="macro")
        else:
            epoch_auc = roc_auc_score(gathered_labels_np, gathered_probs_np, average="macro", multi_class="ovr")
            epoch_map = average_precision_score(gathered_labels_np, gathered_probs_np, average="macro")
    except ValueError:
        epoch_auc = float('nan')
        epoch_map = float('nan')

    global_probs = torch.tensor(gathered_probs_np)
    global_labels = torch.tensor(gathered_labels_np)
    epoch_acc = accuracy_metric(global_probs, global_labels).item()
    
    conf_matrices = []
    if args.task == 'multiclass':
        global_preds = torch.nn.functional.one_hot(global_probs.argmax(dim=1), args.num_classes).float()
        global_labels = torch.nn.functional.one_hot(global_labels, args.num_classes).float()
    else:
        global_preds = (global_probs >= 0.5).float()
    for label_idx in range(global_labels.shape[1]):
        y_true_label = global_labels[:, label_idx].numpy()
        y_pred_label = global_preds[:, label_idx].numpy()
        conf_matrix = confusion_matrix(y_true_label, y_pred_label)
        conf_matrices.append(conf_matrix)
        print(f"[Test] Confusion Matrix for Label {label_idx}:\n{conf_matrix}")
    
    metric_logger.synchronize_between_processes()
    string_1 = "[Test] Averaged stats:{}".format(metric_logger)
    print(string_1)
    string_2 = "[Test] Global stats: epoch_auc={}, epoch_map={}, epoch_acc={}".format(epoch_auc, epoch_map, epoch_acc)
    print(string_2)

    # Save results to a file on rank 0
    if torch.distributed.get_rank() == 0:
        results_path = os.path.join(args.output_dir, "results.txt")
        with open(results_path, 'w') as f:
            f.write(string_1 + '\n')
            f.write(string_2 + '\n')
            for idx, conf_matrix in enumerate(conf_matrices):
                f.write(f"Label {idx}:\n")
                f.write("\n".join(["\t".join(map(str, row)) for row in conf_matrix]) + "\n\n")
            for name, prob, label in zip(decoded_names, gathered_probs_np, gathered_labels_np):
                f.write(f"{name}\t{prob.tolist()}\t{label.tolist()}\n")
        print(f"Results saved to {results_path}")
        
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict['epoch_auc'] = epoch_auc
    return_dict['epoch_map'] = epoch_map
    return_dict['epoch_acc'] = epoch_acc
    return return_dict