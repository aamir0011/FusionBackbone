from __future__ import print_function, division
import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from models.modeling_irene import IRENE, CONFIGS


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IreneDataset(Dataset):
    def __init__(self, pkl_path, img_dir, transform=None):
        with open(pkl_path, 'rb') as f:
            self.mm_data = pickle.load(f)
        self.keys = list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        img_path = os.path.join(self.img_dir, f"{k}.png")
        img = Image.open(img_path).convert('RGB')

        row = self.mm_data[k]
        label = np.asarray(row['label'], dtype=np.float32)
        cc = np.asarray(row['pdesc'], dtype=np.float32)
        demo = np.asarray(row['bics'], dtype=np.float32)
        lab = np.asarray(row['bts'], dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(label), torch.from_numpy(cc), torch.from_numpy(demo), torch.from_numpy(lab)


def compute_auroc(y_true, y_prob):
    y_true = y_true.cpu().numpy()
    y_prob = y_prob.cpu().numpy()
    n_classes = y_true.shape[1]
    aucs = []
    for i in range(n_classes):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue
        aucs.append(roc_auc_score(col, y_prob[:, i]))
    if not aucs:
        return float('nan')
    return float(np.mean(aucs))


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    all_gt, all_pr = [], []
    losses = []

    for imgs, labels, cc, demo, lab in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        cc = cc.view(-1, 40, cc.shape[-1]).to(device, non_blocking=True).float()
        demo = demo.view(-1, 1, demo.shape[-1]).to(device, non_blocking=True).float()
        lab = lab.view(-1, lab.shape[-1], 1).to(device, non_blocking=True).float()
        sex = demo[:, :, 1].view(-1, 1, 1)
        age = demo[:, :, 0].view(-1, 1, 1)

        with autocast(device_type='cuda', enabled=use_amp and device.type == 'cuda'):
            loss = model(imgs, cc, lab, sex, age, labels)
            logits, _, _ = model(imgs, cc, lab, sex, age)
            probs = torch.sigmoid(logits)

        losses.append(loss.item())
        all_gt.append(labels.detach())
        all_pr.append(probs.detach())

    gt = torch.cat(all_gt, dim=0)
    pr = torch.cat(all_pr, dim=0)
    return float(np.mean(losses)), compute_auroc(gt, pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_pkl', required=True)
    ap.add_argument('--val_pkl', required=True)
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='experiments/run1')
    ap.add_argument('--amp', action='store_true', help='Enable mixed precision for faster GPU training')
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = IreneDataset(args.train_pkl, args.img_dir, tx)
    val_ds = IreneDataset(args.val_pkl, args.img_dir, tx)

    # infer num_classes from first sample
    first = train_ds[0]
    num_classes = int(first[1].shape[0])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    scaler = GradScaler('cuda', enabled=args.amp and device.type == 'cuda')

    config = CONFIGS['IRENE']
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_auc = -1.0
    best_path = out_dir / 'best_model.pth'

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for imgs, labels, cc, demo, lab in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            cc = cc.view(-1, 40, cc.shape[-1]).to(device, non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[-1]).to(device, non_blocking=True).float()
            lab = lab.view(-1, lab.shape[-1], 1).to(device, non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1)
            age = demo[:, :, 0].view(-1, 1, 1)

            with autocast(device_type='cuda', enabled=args.amp and device.type == 'cuda'):
                loss = model(imgs, cc, lab, sex, age, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_auc = evaluate(model, val_loader, device, use_amp=args.amp)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_auc': val_auc}, best_path)

    print(f"Done. Best val_auc={best_auc:.4f}, checkpoint={best_path}")


if __name__ == '__main__':
    main()
