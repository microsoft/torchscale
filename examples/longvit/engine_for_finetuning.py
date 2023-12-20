# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import sys
import json
import numpy as np
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import ModelEma
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sksurv.metrics import concordance_index_censored

import utils


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class TCGASubtypingHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        if args.label_smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.probs = []
        self.labels = []

    def train_batch(self, model, image, label):
        logits = model(image=image)
        return {
            "loss": self.criterion(logits, label),
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.probs.clear()
        self.labels.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, label):
        logits = model(image=image)
        probs = F.softmax(logits, dim=1)
        batch_size = image.shape[0]
        acc = (logits.max(-1)[-1] == label).float().sum(0) * 100.0 / batch_size
        self.metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        self.probs.append(probs)
        self.labels.append(label)

    def after_eval(self, data_items, **kwargs):
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.acc))
        result_dict = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        all_probs = torch.cat(self.probs, dim=0)
        all_labels = torch.cat(self.labels, dim=0).tolist()
        n_classes = all_probs.size(-1)
        if n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1].tolist())
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].tolist())
                    print(calc_auc(fpr, tpr))
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    print('nan')
                    aucs.append(float('nan'))

            auc_score = np.nanmean(np.array(aucs))
        
        patient_results = {}
        for index in range(len(all_labels)):
            slide_id = data_items[index]["image_path"]
            assert all_labels[index] == data_items[index]["label"]
            patient_results.update({slide_id: {'prob': all_probs[index, :].tolist(), 'label': all_labels[index]}})

        result_dict["auc"] = auc_score
        print("Acc: {} Auc: {}".format(result_dict["acc"], result_dict["auc"]))
        return result_dict, "auc"


class TCGASurvivalHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        self.criterion = utils.NLLSurvLoss(alpha=0.0)
        self.risk_scores = []
        self.censorships = []
        self.event_times = []
        self.labels = []

    def train_batch(self, model, image, label, event_time, censorship):
        logits = model(image=image)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return {
            "loss": self.criterion(hazards=hazards, S=survival, Y=label, c=censorship),
        }

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.risk_scores.clear()
        self.censorships.clear()
        self.event_times.clear()
        self.labels.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, label, event_time, censorship):
        logits = model(image=image)
        probs = F.softmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        batch_size = image.shape[0]
        acc = (logits.max(-1)[-1] == label).float().sum(0) * 100.0 / batch_size
        self.metric_logger.meters['acc'].update(acc.item(), n=batch_size)

        self.risk_scores.append(risk)
        self.censorships.append(censorship)
        self.event_times.append(event_time)
        self.labels.append(label)

    def after_eval(self, data_items, **kwargs):
        print('* Acc {acc.global_avg:.3f}'.format(acc=self.metric_logger.acc))
        result_dict = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
        all_risk_scores = torch.cat(self.risk_scores, dim=0).cpu().numpy()
        all_censorships = torch.cat(self.censorships, dim=0).cpu().numpy()
        all_event_times = torch.cat(self.event_times, dim=0).cpu().numpy()
        all_labels = torch.cat(self.labels, dim=0).cpu().numpy()
        
        patient_results = {}
        for index in range(len(all_risk_scores)):
            case_id = data_items[index]["case_id"]
            assert int(all_event_times[index]) == int(data_items[index]["event_time"])
            patient_results.update({case_id: {'case_id': case_id, 'risk': all_risk_scores[index], 'disc_label': all_labels[index], 'survival':all_event_times[index], 'censorship':all_censorships[index]}})

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        result_dict["c_index"] = c_index
        print("Acc: {} C_Index: {}".format(result_dict["acc"], result_dict["c_index"]))
        return result_dict, "c_index"


def get_handler(args):
    if args.task.endswith("subtyping"):
        return TCGASubtypingHandler(args)
    elif args.task.endswith("survival"):
        return TCGASurvivalHandler(args)
    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer: Optional[utils.TensorboardLogger] = None, 
        task = None, seq_parallel = False,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if seq_parallel:
            if utils.get_rank() != 0:
                loss = loss * 0.0

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value, 
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval(data_loader.dataset.items)
