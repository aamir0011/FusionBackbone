import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# IRENE class order from irene.py
DISEASES = [
    'COPD', 'Bronchiectasis', 'Pneumothorax', 'Pneumonia',
    'ILD', 'Tuberculosis', 'Lung cancer', 'Pleural effusion'
]


def map_finding_to_label(finding: str):
    y = np.zeros((8,), dtype=np.float32)
    f = (finding or '').lower()
    if 'pneumonia' in f or 'covid' in f:
        y[3] = 1.0  # Pneumonia
    if 'pneumothorax' in f:
        y[2] = 1.0
    if 'tuberculosis' in f:
        y[5] = 1.0
    if 'cancer' in f or 'neoplasm' in f:
        y[6] = 1.0
    if 'effusion' in f:
        y[7] = 1.0
    if 'copd' in f:
        y[0] = 1.0
    if 'bronchiectasis' in f:
        y[1] = 1.0
    if 'interstitial' in f or 'ild' in f:
        y[4] = 1.0
    if y.sum() == 0:
        y[3] = 1.0  # fallback
    return y


def sex_to_num(sex):
    if sex is None:
        return 0.5
    s = str(sex).strip().lower()
    if s.startswith('m'):
        return 1.0
    if s.startswith('f'):
        return 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='path to covid-chestxray-dataset root')
    ap.add_argument('--out-data-dir', required=True, help='folder where png images will be written')
    ap.add_argument('--out-pkl-prefix', required=True, help='output prefix for pkl (without .pkl)')
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    metadata = pd.read_csv(src / 'metadata.csv')
    metadata = metadata[metadata['filename'].notna()].copy()
    metadata = metadata.sample(frac=1.0, random_state=args.seed).head(args.limit)

    out_dir = Path(args.out_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = {}
    written = 0

    for _, row in metadata.iterrows():
        folder = str(row.get('folder', '') or '')
        filename = str(row['filename'])
        src_img = src / folder / filename
        if not src_img.exists():
            continue

        case_id = Path(filename).stem.replace(' ', '_')
        out_img = out_dir / f'{case_id}.png'

        try:
            img = Image.open(src_img).convert('RGB')
            img.save(out_img)
        except Exception:
            continue

        age = row.get('age', np.nan)
        age = float(age) if pd.notna(age) else 50.0
        sex = sex_to_num(row.get('sex', ''))

        # NOTE: Original paper uses rich text/lab embeddings. Here we create
        # compatible placeholder tensors so the pipeline can run on a similar dataset.
        pdesc = np.zeros((1, 40, 768), dtype=np.float32)
        bics = np.array([age, sex], dtype=np.float32)
        bts = np.zeros((92,), dtype=np.float32)
        label = map_finding_to_label(str(row.get('finding', '')))

        subset[case_id] = {
            'pdesc': pdesc,
            'bics': bics,
            'bts': bts,
            'label': label,
        }
        written += 1

    out_pkl = Path(f"{args.out_pkl_prefix}.pkl")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(subset, f)

    print(f'Wrote {written} images to: {out_dir}')
    print(f'Wrote metadata pkl: {out_pkl}')


if __name__ == '__main__':
    main()
