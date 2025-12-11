#!/usr/bin/env python3
"""
YOLO v11 Training Script.

Usage:
    python train.py --task detect --model n --epochs 300 --batch-size 16
    python train.py --task segment --model s --epochs 300 --data coco.yaml
    python train.py --task pose --model m --epochs 300 --data coco-pose.yaml
"""

import argparse
import copy
import csv
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.models import yolo_v11_n, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x
from yolo.utils.loss import ComputeLoss
from yolo.utils.training import (
    setup_seed,
    setup_multi_processes,
    EMA,
    LinearLR,
    AverageMeter,
    set_params,
    strip_optimizer,
)
from yolo.data.dataset import DetectionDataset, create_dataloader

warnings.filterwarnings("ignore")


MODEL_FACTORY = {
    'n': yolo_v11_n,
    's': yolo_v11_s,
    'm': yolo_v11_m,
    'l': yolo_v11_l,
    'x': yolo_v11_x,
}


def train(args, params):
    """Main training loop."""
    
    # Create model
    model_fn = MODEL_FACTORY[args.model]
    model = model_fn(len(params['names']), task=args.task)
    model.cuda()
    
    print(f"Model: YOLOv11-{args.model.upper()} for {args.task}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer setup
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(
        set_params(model, params['weight_decay']),
        params['min_lr'],
        params['momentum'],
        nesterov=True
    )

    # EMA
    ema = EMA(model) if args.local_rank == 0 else None

    # Dataset
    train_files = load_dataset_files(args.data_dir, 'train')
    dataset = DetectionDataset(train_files, args.input_size, params, augment=True)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=8
    )
    loader.sampler = sampler

    # Scheduler
    num_steps = len(loader)
    scheduler = LinearLR(args, params, num_steps)

    # DDP
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    best_map = 0
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = ComputeLoss(model, params)

    # Training loop
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=[
                'epoch', 'box', 'cls', 'dfl', 'Recall', 'Precision', 'mAP@50', 'mAP'
            ])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            
            if args.distributed:
                sampler.set_epoch(epoch)
            
            # Disable mosaic for last 10 epochs
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            pbar = enumerate(loader)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                pbar = tqdm.tqdm(pbar, total=num_steps)

            optimizer.zero_grad()
            avg_box = AverageMeter()
            avg_cls = AverageMeter()
            avg_dfl = AverageMeter()

            for i, (samples, targets) in pbar:
                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                # Forward
                with torch.cuda.amp.autocast():
                    outputs = model(samples)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box.update(loss_box.item(), samples.size(0))
                avg_cls.update(loss_cls.item(), samples.size(0))
                avg_dfl.update(loss_dfl.item(), samples.size(0))

                # Scale losses
                loss_box *= args.batch_size * args.world_size
                loss_cls *= args.batch_size * args.world_size
                loss_dfl *= args.batch_size * args.world_size

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                    s = ('%10s' * 2 + '%10.3g' * 3) % (
                        f'{epoch + 1}/{args.epochs}', memory,
                        avg_box.avg, avg_cls.avg, avg_dfl.avg
                    )
                    pbar.set_description(s)

            # Validation
            if args.local_rank == 0:
                metrics = validate(args, params, ema.ema)

                logger.writerow({
                    'epoch': str(epoch + 1).zfill(3),
                    'box': f'{avg_box.avg:.3f}',
                    'cls': f'{avg_cls.avg:.3f}',
                    'dfl': f'{avg_dfl.avg:.3f}',
                    'mAP': f'{metrics[0]:.3f}',
                    'mAP@50': f'{metrics[1]:.3f}',
                    'Recall': f'{metrics[2]:.3f}',
                    'Precision': f'{metrics[3]:.3f}',
                })
                log.flush()

                # Save
                if metrics[0] > best_map:
                    best_map = metrics[0]

                save = {
                    'epoch': epoch + 1,
                    'model': copy.deepcopy(ema.ema),
                }
                torch.save(save, 'weights/last.pt')
                if best_map == metrics[0]:
                    torch.save(save, 'weights/best.pt')

    if args.local_rank == 0:
        strip_optimizer('weights/best.pt')
        strip_optimizer('weights/last.pt')

    torch.cuda.empty_cache()


@torch.no_grad()
def validate(args, params, model=None):
    """Validation loop."""
    val_files = load_dataset_files(args.data_dir, 'val')
    dataset = DetectionDataset(val_files, args.input_size, params, augment=False)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=4)

    if model is None:
        checkpoint = torch.load('weights/best.pt', map_location='cuda')
        model = checkpoint['model'].float().fuse()

    model.half()
    model.eval()

    from yolo.utils.boxes import non_max_suppression, wh2xy
    from yolo.utils.metrics import MetricTracker

    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()
    tracker = MetricTracker(len(params['names']))

    pbar = tqdm.tqdm(loader, desc='%10s' * 5 % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    
    for samples, targets in pbar:
        samples = samples.cuda().half() / 255.0
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h)).cuda()

        outputs = model(samples)
        outputs = non_max_suppression(outputs)

        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx].cuda()
            box = targets['box'][idx].cuda()

            if cls.shape[0]:
                target = torch.cat((cls, wh2xy(box) * scale), dim=1)
                tracker.update(output[:, :6], target)
            elif output.shape[0]:
                tracker.update(output[:, :6], torch.zeros((0, 5)).cuda())

    m_pre, m_rec, map50, mean_ap = tracker.compute()
    print('%10s' + '%10.3g' * 4 % ('', m_pre, m_rec, map50, mean_ap))
    
    model.float()
    return mean_ap, map50, m_rec, m_pre


def load_dataset_files(data_dir: str, split: str) -> list:
    """Load dataset file paths."""
    txt_file = os.path.join(data_dir, f'{split}2017.txt')
    
    if not os.path.exists(txt_file):
        # Try alternative naming
        txt_file = os.path.join(data_dir, f'{split}.txt')
    
    files = []
    if os.path.exists(txt_file):
        with open(txt_file) as f:
            for line in f:
                path = line.strip()
                if os.path.isabs(path):
                    files.append(path)
                else:
                    files.append(os.path.join(data_dir, 'images', split + '2017', os.path.basename(path)))
    else:
        # Fallback: scan directory
        img_dir = os.path.join(data_dir, 'images', split + '2017')
        if os.path.exists(img_dir):
            for ext in ('jpg', 'jpeg', 'png'):
                files.extend([str(p) for p in Path(img_dir).glob(f'*.{ext}')])
    
    return files


def main():
    parser = argparse.ArgumentParser(description='YOLO v11 Training')
    
    # Model
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--task', default='detect', choices=['detect', 'segment', 'pose'],
                        help='Task type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    
    # Data
    parser.add_argument('--data-dir', default='../Dataset/COCO', help='Dataset directory')
    parser.add_argument('--config', default='configs/default.yaml', help='Config file')
    
    # Distributed
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup distributed
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        os.makedirs('weights', exist_ok=True)

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    
    with open(config_path) as f:
        params = yaml.safe_load(f)

    setup_seed()
    setup_multi_processes()

    if args.local_rank == 0:
        print(f"Task: {args.task}")
        print(f"Model: YOLOv11-{args.model.upper()}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Input size: {args.input_size}")

    train(args, params)


if __name__ == '__main__':
    main()

