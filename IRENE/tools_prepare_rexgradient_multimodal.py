import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
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


def parse_age(x):
    s = str(x or '').strip().upper().replace('Y', '')
    return float(s) if s.isdigit() else 50.0


def parse_sex(x):
    s = str(x or '').strip().upper()
    if s.startswith('M'):
        return 1.0
    if s.startswith('F'):
        return 0.0
    return 0.5


def prep_split(ds, out_img_dir: Path, out_pkl: Path, limit=None):
    n = len(ds) if limit is None else min(limit, len(ds))
    d = {}

    # dataset is report-only; create deterministic placeholder image token path per sample
    blank_path = out_img_dir / '_blank.png'
    if not blank_path.exists():
        Image.new('RGB', (224, 224), color=(128, 128, 128)).save(blank_path)

    for i in range(n):
        r = ds[i]
        cid = f"case_{i:07d}"
        p = out_img_dir / f"{cid}.png"
        if not p.exists():
            p.symlink_to(blank_path.name)

        text = ' '.join([
            str(r.get('Indication', '') or ''),
            str(r.get('Comparison', '') or ''),
            str(r.get('Findings', '') or ''),
            str(r.get('Impression', '') or ''),
        ]).strip()

        d[cid] = {
            'pdesc': text_to_pdesc(text),
            'bics': np.array([parse_age(r.get('PatientAge')), parse_sex(r.get('PatientSex'))], dtype=np.float32),
            'bts': np.zeros((92,), dtype=np.float32),
            'label': map_to_label(text),
        }

    with open(out_pkl, 'wb') as f:
        pickle.dump(d, f)
    print(f"saved {len(d)} -> {out_pkl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_data_dir', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--train_n', type=int, default=12000)
    ap.add_argument('--val_n', type=int, default=2000)
    ap.add_argument('--test_n', type=int, default=2000)
    args = ap.parse_args()

    ds = load_dataset('rajpurkarlab/ReXGradient-160K')
    out = Path(args.out_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    prep_split(ds['train'], out, Path(f"{args.prefix}_train.pkl"), args.train_n)
    prep_split(ds['validation'], out, Path(f"{args.prefix}_val.pkl"), args.val_n)
    prep_split(ds['test'], out, Path(f"{args.prefix}_test.pkl"), args.test_n)


if __name__ == '__main__':
    main()
