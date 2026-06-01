import argparse
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image


DISEASES = [
    'COPD', 'Bronchiectasis', 'Pneumothorax', 'Pneumonia',
    'ILD', 'Tuberculosis', 'Lung cancer', 'Pleural effusion'
]


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


def map_to_label(question: str, answer: str):
    s = f"{question} {answer}".lower()
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
    if 'effusion' in s:
        y[7] = 1
    if y.sum() == 0:
        y[3] = 1
    return y


def parse_demo(text: str):
    s = text.lower()
    sex = 0.5
    if ' male' in s or s.startswith('male'):
        sex = 1.0
    elif ' female' in s or s.startswith('female'):
        sex = 0.0

    age = 50.0
    for tok in s.replace(',', ' ').split():
        if tok.isdigit():
            v = int(tok)
            if 0 < v < 120:
                age = float(v)
                break
    return np.array([age, sex], dtype=np.float32)


def make_case_id(split, idx):
    return f"{split}_{idx:08d}"


def save_split(ds_split, split_name, out_img_dir, out_pkl_path, limit=None):
    subset = {}
    n = len(ds_split) if limit is None else min(limit, len(ds_split))

    for i in range(n):
        row = ds_split[i]
        case_id = make_case_id(split_name, i)

        img = row.get('image', None)
        try:
            if isinstance(img, Image.Image):
                img_pil = img.convert('RGB')
            else:
                # some datasets keep only path-like fields
                path = row.get('location') or row.get('path')
                if not path or not os.path.exists(path):
                    continue
                img_pil = Image.open(path).convert('RGB')

            img_pil.save(out_img_dir / f"{case_id}.png")
        except Exception:
            continue

        q = str(row.get('question', ''))
        a = str(row.get('answer', ''))
        merged_text = f"Question: {q}. Answer: {a}."

        subset[case_id] = {
            'pdesc': text_to_pdesc(merged_text),
            'bics': parse_demo(merged_text),
            'bts': np.zeros((92,), dtype=np.float32),
            'label': map_to_label(q, a),
        }

    with open(out_pkl_path, 'wb') as f:
        pickle.dump(subset, f)
    print(f"Saved {len(subset)} samples -> {out_pkl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_disk_path', required=True, help='path from datasets.save_to_disk')
    ap.add_argument('--out_data_dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--train_limit', type=int, default=None)
    ap.add_argument('--val_limit', type=int, default=None)
    ap.add_argument('--test_limit', type=int, default=None)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_disk_path)
    out_data_dir = Path(args.out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    if 'train' in ds:
        save_split(ds['train'], 'train', out_data_dir, Path(f"{args.prefix}_train.pkl"), args.train_limit)
    if 'validation' in ds:
        save_split(ds['validation'], 'val', out_data_dir, Path(f"{args.prefix}_val.pkl"), args.val_limit)
    elif 'test' in ds:
        # fallback: use test as val when validation split not available
        save_split(ds['test'], 'val', out_data_dir, Path(f"{args.prefix}_val.pkl"), args.val_limit)

    if 'test' in ds:
        save_split(ds['test'], 'test', out_data_dir, Path(f"{args.prefix}_test.pkl"), args.test_limit)


if __name__ == '__main__':
    main()
