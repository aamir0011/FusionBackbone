#!/usr/bin/env python3
import csv, subprocess, traceback
from pathlib import Path

REPO=Path('/home/aamir/harbor/workspaces/yousuf/thesis')
VENV='. .venv/bin/activate'
EPOCHS=50
DATASETS=[
 ('vqarad','data/vqa_rad_train.pkl','data/vqa_rad_val.pkl','data/vqa_rad_images'),
 ('pathvqa','data/path_vqa_train.pkl','data/path_vqa_val.pkl','data/path_vqa_images'),
 ('mimic','data/mimic_train.pkl','data/mimic_val.pkl','data/mimic_images'),
 ('rexgradient','data/rexgradient_train.pkl','data/rexgradient_val.pkl','data/rexgradient_images'),
]

def sh(cmd):
    p=subprocess.run(cmd,shell=True,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode!=0:
        raise RuntimeError(f"{cmd}\n{p.stdout}")
    return p.stdout

def main():
    out=REPO/'experiments'/'phaseA_shaspec_gating_50ep'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ds,tr,va,img in DATASETS:
        od=out/f'shaspec_gating_{ds}'
        od.mkdir(parents=True,exist_ok=True)
        sm=out/f'shaspec_gating_{ds}_summary.csv'
        cmd=(f"{VENV} && cd IRENE && python train_shaspec_gating.py "
             f"--train_pkl {tr} --val_pkl {va} --img_dir {img} --epochs {EPOCHS} --batch 12 --num_workers 0 "
             f"--out_dir ../{od.relative_to(REPO)} --summary_csv ../{sm.relative_to(REPO)}")
        log=sh(cmd)
        (od/'train.log').write_text(log)
        srow=list(csv.DictReader(open(sm)))[0]
        rows.append({'dataset':ds,'epochs':EPOCHS,'best_val_auc':srow['best_val_auc'],'last_val_loss':srow['last_val_loss'],'run_dir':str(od.relative_to(REPO))})

    out_csv=out/'phaseA_shaspec_gating_50ep_all_datasets.csv'
    with open(out_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','epochs','best_val_auc','last_val_loss','run_dir'])
        w.writeheader(); w.writerows(rows)
    print('PHASEA_SHASPEC_DONE')
    print(out_csv)

if __name__=='__main__':
    try:
        main()
    except Exception:
        err=REPO/'experiments'/'phaseA_shaspec_gating_50ep'/'run_error.log'
        err.parent.mkdir(parents=True,exist_ok=True)
        err.write_text(traceback.format_exc())
        raise
