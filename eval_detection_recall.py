# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Suchen for HOI detection

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate, get_flop_stats
from models import build_model
from arguments import get_args_parser
from utils.scheduler import create_scheduler


def filter_detections(detection_res, score_threshold=0.4):
    bboxes, scores = detection_res
    bboxes = torch.as_tensor(bboxes)
    scores = torch.as_tensor(scores)
    keep_idx = scores > score_threshold
    bboxes, scores = bboxes[keep_idx], scores[keep_idx]
    return bboxes, scores

def calculate_recall(gt_boxes, pred_boxes, threshold=0.5):
    '''
    gt_boxes is a list of ground truth bounding boxes, and pred_boxes is a list of predicted bounding boxes.
    The function iterates over each predicted box and calculates the Intersection over Union (IoU) with each ground truth box.
    '''
    num_gt_boxes = len(gt_boxes)
    true_positives = 0

    for gt_box in gt_boxes:
        ious = [calculate_iou(gt_box, pred_box) for pred_box in pred_boxes]
        if len(ious) == 0:
            continue
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)
        del pred_boxes[max_iou_idx]
        
        if max_iou >= threshold:
            true_positives += 1

    recall = true_positives / num_gt_boxes
    return recall

def calculate_iou(boxA, boxB):
    '''
    boxA and boxB represent two bounding boxes in the format [x_min, y_min, x_max, y_max].
    The function calculates the coordinates of the intersection rectangle (if it exists) and computes the IoU as the ratio of the intersection area to the union area.
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

def main(args):
    """Training and evaluation function"""

    # distributed data parallel setup
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    # print(args)

    # device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    # model.to(device)

    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    # # optimizer setup
    # def build_optimizer(model):
    #     # * frozen CLIP model
    #     update_modules, update_params = [], []
    #     frozen_modules, frozen_params = [], []
    #     for n, p in model.named_parameters():
    #         if 'hoi' in n or 'bbox' in n or 'promp_proj' in n:
    #             update_modules.append(n)
    #             update_params.append(p)
    #         else:
    #             frozen_modules.append(n)
    #             frozen_params.append(p)
    #             p.requires_grad = False

    #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    #                                   lr=args.lr, weight_decay=args.weight_decay)
    #     return optimizer

    # optimizer = build_optimizer(model_without_ddp)
    # lr_scheduler, _ = create_scheduler(args, optimizer)

    # n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    # print('number of trainable params:', n_parameters, f'{n_parameters/1e6:.3f}M')

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print('# train:', len(dataset_train), ', # val', len(dataset_val))

    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    import pickle
    pkl_path = f"/network_space/storage43/lttt/swig_hoi/mdef_detr/all_objects.pkl"
    with open(pkl_path,'rb') as f:
        pred_detections = pickle.load(f)
    
    import utils.box_ops as box_ops
    from tqdm import tqdm
    all_recall = []
    for i, (image, targets) in enumerate(tqdm(data_loader_train)):
        ori_h, ori_w = targets[0]['orig_size'] # "orig_size": torch.tensor([h, w]),
        gt_boxes = targets[0]['boxes']
        if len(gt_boxes) == 0:
            print("[WARNING] no annotation!!", targets[0])
            continue
        gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes[:, 0::2] = gt_boxes[:, 0::2] * ori_w
        gt_boxes[:, 1::2] = gt_boxes[:, 1::2] * ori_h
        gt_boxes_list = gt_boxes.int().tolist()
        filename = targets[0]['filename'].split("/")[-1].split(".")[0]
        pred_boxes, pred_scores = filter_detections(pred_detections[filename], score_threshold=0.1)
        pred_boxes_list = pred_boxes.tolist()
        recall = calculate_recall(gt_boxes=gt_boxes_list, pred_boxes=pred_boxes_list, threshold=0.5)
        all_recall.append(recall)
        if i % 1000 == 0:
            print("average recall:", np.mean(all_recall))
    
    import pdb; pdb.set_trace()
    # resume from the given checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print(f"load checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # evaluation
    if args.eval:
        # print FLOPs
        # get_flop_stats(model, data_loader_val)
        if args.distributed:
            model = model.module
        test_stats, evaluator = evaluate(model, postprocessors, criterion, data_loader_val, device, args)
        if args.output_dir:
            evaluator.save(args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer,
                                      device, epoch, args.clip_max_norm)
        lr_scheduler.step(epoch)

        # checkpoint saving
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)