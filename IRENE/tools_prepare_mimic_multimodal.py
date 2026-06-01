import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image


def token_vec(token: str, dim=768):
    h = hashlib.sha256(token.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], 'big') % (2**32 - 1)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.02, size=(dim,)).astype(np.float32)


def text_to_pdesc(text: str, tk_lim=40, dim=768):
    tokens = [t for t in text.lower().replace('\n', ' ').split(' ') if t][:tk_lim]
    arr = np.zeros((1, tk_lim, dim), dtype=np.float32)
    for i, t in enumerate(tokens):
        arr[0, i] = token_vec(t, dim)
    return arr


def map_to_label(text: str):
    s = text.lower()
    y = np.zeros((8,), dtype=np.float32)
    if 'copd' in s:
        y[0] = 1
    if 'bronchiectasis' in s:
        y[1] = 1
    if 'pneumothorax' in s:
        y[2] = 1
    if 'pneumonia' in s or 'covid' in s:
        y[3] = 1
    if 'interstitial' in s or 'ild' in s:
        y[4] = 1
    if 'tuberculosis' in s or 'tb' in s:
        y[5] = 1
    if 'cancer' in s or 'tumor' in s or 'carcinoma' in s:
        y[6] = 1
    if 'effusion' in s or 'pleural effusion' in s:
        y[7] = 1
    if y.sum() == 0:
        y[3] = 1
    return y


def save_subset(rows, out_img_dir, out_pkl):
    d = {}
    for idx, row in enumerate(rows):
        case_id = f"case_{idx:07d}"
        try:
            img = row['image']
            if not isinstance(img, Image.Image):
                continue
            img.convert('RGB').save(out_img_dir / f"{case_id}.png")
        except Exception:
            continue

        findings = str(row.get('findings', '') or '')
        impression = str(row.get('impression', '') or '')
        text = (findings + ' ' + impression).strip()

        d[case_id] = {
            'pdesc': text_to_pdesc(text),
            'bics': np.array([50.0, 0.5], dtype=np.float32),
            'bts': np.zeros((92,), dtype=np.float32),
            'label': map_to_label(text),
        }

    with open(out_pkl, 'wb') as f:
        pickle.dump(d, f)
    print(f"saved {len(d)} -> {out_pkl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_disk_path', required=True)
    ap.add_argument('--out_data_dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--train_n', type=int, default=20000)
    ap.add_argument('--val_n', type=int, default=5000)
    ap.add_argument('--test_n', type=int, default=5000)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_disk_path)['train']
    out_data_dir = Path(args.out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    total = min(len(ds), args.train_n + args.val_n + args.test_n)
    rows = [ds[i] for i in range(total)]
    train = rows[:args.train_n]
    val = rows[args.train_n:args.train_n + args.val_n]
    test = rows[args.train_n + args.val_n:args.train_n + args.val_n + args.test_n]

    save_subset(train, out_data_dir, Path(f"{args.prefix}_train.pkl"))
    save_subset(val, out_data_dir, Path(f"{args.prefix}_val.pkl"))
    save_subset(test, out_data_dir, Path(f"{args.prefix}_test.pkl"))


if __name__ == '__main__':
    main()
