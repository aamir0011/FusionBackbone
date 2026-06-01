from __future__ import annotations
import argparse,csv,os,pickle,random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from models.modeling_irene import IRENE, CONFIGS
TK=40

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
class DS(Dataset):
    def __init__(self,pkl,img,tx): self.d=pickle.load(open(pkl,'rb')); self.k=list(self.d.keys()); self.img=img; self.tx=tx
    def __len__(self): return len(self.k)
    def __getitem__(self,i):
        k=self.k[i]; r=self.d[k]; im=self.tx(Image.open(os.path.join(self.img,f"{k}.png")).convert('RGB'))
        cc=np.asarray(r['pdesc'],np.float32); cc=cc[0] if cc.ndim==3 else cc
        if cc.shape[0]<TK: cc=np.concatenate([cc,np.zeros((TK-cc.shape[0],cc.shape[1]),np.float32)],0)
        return im,torch.tensor(np.asarray(r['label'],np.float32)),torch.tensor(cc[:TK]),torch.tensor(np.asarray(r['bics'],np.float32)),torch.tensor(np.asarray(r['bts'],np.float32))
class Gater(nn.Module):
    def __init__(self):
        super().__init__(); self.fc=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,3),nn.Sigmoid())
    def forward(self,pres):
        g=self.fc(pres)
        return g[:,0].view(-1,1,1,1), g[:,1].view(-1,1,1), g[:,2].view(-1,1,1)
def auc(y,p):
    y,p=y.cpu().numpy(),p.cpu().numpy(); z=[]
    for i in range(y.shape[1]):
        if len(np.unique(y[:,i]))>1: z.append(roc_auc_score(y[:,i],p[:,i]))
    return float(np.mean(z)) if z else float('nan')
def run(a):
    seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tx=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    tr,va=DS(a.train_pkl,a.img_dir,tx),DS(a.val_pkl,a.img_dir,tx); tr.k=tr.k[:a.max_train]; va.k=va.k[:a.max_val]
    trl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=2); vl=DataLoader(va,batch_size=a.batch,shuffle=False,num_workers=2)
    m=IRENE(CONFIGS['IRENE'],224,zero_head=True,num_classes=8).to(dev); g=Gater().to(dev)
    opt=torch.optim.AdamW(list(m.parameters())+list(g.parameters()),lr=a.lr); bce=nn.BCEWithLogitsLoss(); rows=[]
    for e in range(1,a.epochs+1):
        m.train(); g.train(); tl=[]
        for img,y,cc,demo,lab in trl:
            img,y,cc,demo,lab=img.to(dev),y.to(dev),cc.to(dev),demo.to(dev),lab.to(dev)
            sex=demo[:,1].view(-1,1,1); age=demo[:,0].view(-1,1,1); lab=lab.view(-1,lab.shape[-1],1)
            b=img.size(0)
            pres=torch.stack([(torch.rand(b,device=dev)>a.miss_p).float(),(torch.rand(b,device=dev)>a.miss_p).float(),(torch.rand(b,device=dev)>a.miss_p).float()],dim=1)
            gi,gt,gs=g(pres)
            logits,_,_=m(img*gi,cc*gt,lab*gs,sex*gs,age*gs); loss=bce(logits,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl.append(loss.item())
        m.eval(); yl=[];pl=[];vloss=[]
        with torch.no_grad():
            for img,y,cc,demo,lab in vl:
                img,y,cc,demo,lab=img.to(dev),y.to(dev),cc.to(dev),demo.to(dev),lab.to(dev)
                sex=demo[:,1].view(-1,1,1); age=demo[:,0].view(-1,1,1); lab=lab.view(-1,lab.shape[-1],1)
                pres=torch.ones(img.size(0),3,device=dev)
                gi,gt,gs=g(pres)
                logits,_,_=m(img*gi,cc*gt,lab*gs,sex*gs,age*gs); vloss.append(bce(logits,y).item()); yl.append(y); pl.append(torch.sigmoid(logits))
        rows.append({'epoch':e,'train_loss':float(np.mean(tl)),'val_loss':float(np.mean(vloss)),'val_auc':auc(torch.cat(yl),torch.cat(pl))}); print(rows[-1])
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    with open(out/'metrics.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(a.summary_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['technique','best_val_auc','last_val_loss']); w.writeheader(); w.writerow({'technique':'modality_wise_gating','best_val_auc':max(r['val_auc'] for r in rows),'last_val_loss':rows[-1]['val_loss']})
if __name__=='__main__':
    p=argparse.ArgumentParser();
    p.add_argument('--train_pkl',required=True); p.add_argument('--val_pkl',required=True); p.add_argument('--img_dir',required=True)
    p.add_argument('--out_dir',required=True); p.add_argument('--summary_csv',required=True)
    p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=8); p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--max_train',type=int,default=64); p.add_argument('--max_val',type=int,default=64); p.add_argument('--miss_p',type=float,default=0.3); p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
