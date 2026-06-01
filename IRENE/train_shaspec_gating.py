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

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self,pkl,img,tx):
        self.d=pickle.load(open(pkl,'rb')); self.k=list(self.d.keys()); self.img=img; self.tx=tx
    def __len__(self): return len(self.k)
    def __getitem__(self,i):
        k=self.k[i]; r=self.d[k]
        im=self.tx(Image.open(os.path.join(self.img,f"{k}.png")).convert('RGB'))
        cc=np.asarray(r['pdesc'],np.float32); cc=cc[0] if cc.ndim==3 else cc
        if cc.shape[0]<TK: cc=np.concatenate([cc,np.zeros((TK-cc.shape[0],cc.shape[1]),np.float32)],0)
        return im,torch.tensor(np.asarray(r['label'],np.float32)),torch.tensor(cc[:TK]),torch.tensor(np.asarray(r['bics'],np.float32)),torch.tensor(np.asarray(r['bts'],np.float32))

class Gater(nn.Module):
    def __init__(self):
        super().__init__(); self.fc=nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,3),nn.Sigmoid())
    def forward(self,p):
        g=self.fc(p)
        return g[:,0].view(-1,1,1,1), g[:,1].view(-1,1,1), g[:,2].view(-1,1,1)

class ShaSpecHeads(nn.Module):
    def __init__(self, d=128, spec_d=128):
        super().__init__()
        # shared encoders
        self.img_shared = nn.Sequential(nn.Linear(3,d), nn.ReLU(), nn.Linear(d,d))
        self.txt_shared = nn.Sequential(nn.Linear(768,d), nn.ReLU(), nn.Linear(d,d))
        self.str_shared = nn.Sequential(nn.Linear(94,d), nn.ReLU(), nn.Linear(d,d))
        # specific encoders
        self.img_spec = nn.Sequential(nn.Linear(3,spec_d), nn.ReLU(), nn.Linear(spec_d,spec_d))
        self.txt_spec = nn.Sequential(nn.Linear(768,spec_d), nn.ReLU(), nn.Linear(spec_d,spec_d))
        self.str_spec = nn.Sequential(nn.Linear(94,spec_d), nn.ReLU(), nn.Linear(spec_d,spec_d))
        self.domain = nn.Linear(spec_d,3)

    def forward(self,img,cc,age,sex,lab):
        # cheap modality summaries
        img_s = img.mean(dim=[2,3])           # [B,3]
        txt_s = cc.mean(dim=1)                # [B,768]
        str_s = torch.cat([age.view(-1,1),sex.view(-1,1),lab.squeeze(-1)],dim=1)  # [B,94]

        sh_i, sh_t, sh_s = self.img_shared(img_s), self.txt_shared(txt_s), self.str_shared(str_s)
        sp_i, sp_t, sp_s = self.img_spec(img_s), self.txt_spec(txt_s), self.str_spec(str_s)
        return (sh_i,sh_t,sh_s),(sp_i,sp_t,sp_s)

def auc(y,p):
    y,p=y.cpu().numpy(),p.cpu().numpy(); z=[]
    for i in range(y.shape[1]):
        if len(np.unique(y[:,i]))>1: z.append(roc_auc_score(y[:,i],p[:,i]))
    return float(np.mean(z)) if z else float('nan')

def alignment_loss(shared, pres):
    sh_i,sh_t,sh_s = shared
    p_i,p_t,p_s = pres[:,0],pres[:,1],pres[:,2]
    loss = 0.0; cnt = 0
    pairs=[(sh_i,sh_t,p_i,p_t),(sh_i,sh_s,p_i,p_s),(sh_t,sh_s,p_t,p_s)]
    for a,b,pa,pb in pairs:
        m = (pa*pb).view(-1,1)
        if m.sum().item()>0:
            loss = loss + (((a-b).pow(2))*m).sum()/(m.sum()*a.size(1)+1e-8)
            cnt += 1
    return loss/cnt if cnt>0 else sh_i.new_tensor(0.0)

def domain_loss(spec, pres, dom_head):
    sp_i,sp_t,sp_s = spec
    logits=[]; targets=[]
    mods=[sp_i,sp_t,sp_s]
    for idx,sp in enumerate(mods):
        m=pres[:,idx]>0.5
        if m.any():
            logits.append(dom_head(sp[m]))
            targets.append(torch.full((int(m.sum().item()),),idx,device=sp.device,dtype=torch.long))
    if not logits:
        return sp_i.new_tensor(0.0)
    return nn.CrossEntropyLoss()(torch.cat(logits,0), torch.cat(targets,0))

def orth_loss(shared,spec):
    out=0.0
    for sh,sp in zip(shared,spec):
        shn = nn.functional.normalize(sh,dim=1)
        spn = nn.functional.normalize(sp,dim=1)
        out = out + (shn*spn).sum(dim=1).abs().mean()
    return out/3.0

def run(a):
    seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tx=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    tr,va=DS(a.train_pkl,a.img_dir,tx),DS(a.val_pkl,a.img_dir,tx)
    trl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=a.num_workers)
    vl=DataLoader(va,batch_size=a.batch,shuffle=False,num_workers=a.num_workers)

    m=IRENE(CONFIGS['IRENE'],224,zero_head=True,num_classes=8).to(dev)
    g=Gater().to(dev)
    h=ShaSpecHeads(d=a.shared_d,spec_d=a.spec_d).to(dev)

    opt=torch.optim.AdamW(list(m.parameters())+list(g.parameters())+list(h.parameters()),lr=a.lr,weight_decay=1e-4)
    bce=nn.BCEWithLogitsLoss(); rows=[]; best=-1
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    for e in range(1,a.epochs+1):
        m.train(); g.train(); h.train(); tl=[]
        for img,y,cc,demo,lab in trl:
            img,y,cc,demo,lab=img.to(dev),y.to(dev),cc.to(dev),demo.to(dev),lab.to(dev)
            sex=demo[:,1].view(-1,1,1); age=demo[:,0].view(-1,1,1); lab=lab.view(-1,lab.shape[-1],1)
            b=img.size(0)
            pres=torch.stack([
                (torch.rand(b,device=dev)>a.miss_p).float(),
                (torch.rand(b,device=dev)>a.miss_p).float(),
                (torch.rand(b,device=dev)>a.miss_p).float()
            ],dim=1)
            gi,gt,gs=g(pres)

            logits,_,_=m(img*gi,cc*gt,lab*gs,sex*gs,age*gs)
            shared,spec = h(img,cc,age,sex,lab)
            l_cls=bce(logits,y)
            l_align=alignment_loss(shared,pres)
            l_dom=domain_loss(spec,pres,h.domain)
            l_orth=orth_loss(shared,spec)
            loss=l_cls + a.align_w*l_align + a.domain_w*l_dom + a.orth_w*l_orth

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl.append(loss.item())

        m.eval(); g.eval(); h.eval(); yl=[];pl=[];vloss=[]
        with torch.no_grad():
            for img,y,cc,demo,lab in vl:
                img,y,cc,demo,lab=img.to(dev),y.to(dev),cc.to(dev),demo.to(dev),lab.to(dev)
                sex=demo[:,1].view(-1,1,1); age=demo[:,0].view(-1,1,1); lab=lab.view(-1,lab.shape[-1],1)
                pres=torch.ones(img.size(0),3,device=dev)
                gi,gt,gs=g(pres)
                logits,_,_=m(img*gi,cc*gt,lab*gs,sex*gs,age*gs)
                vloss.append(bce(logits,y).item()); yl.append(y); pl.append(torch.sigmoid(logits))

        row={'epoch':e,'train_loss':float(np.mean(tl)),'val_loss':float(np.mean(vloss)),'val_auc':auc(torch.cat(yl),torch.cat(pl))}
        rows.append(row); print(row)
        if np.isfinite(row['val_auc']) and row['val_auc']>best:
            best=row['val_auc']
            torch.save({'model':m.state_dict(),'gater':g.state_dict(),'shaspec':h.state_dict(),'val_auc':best,'epoch':e}, out/'best_model.pth')

    with open(out/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(a.summary_csv,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['technique','dataset','best_val_auc','last_val_loss','epochs','align_w','domain_w','orth_w'])
        w.writeheader(); w.writerow({'technique':'shaspec_gating','dataset':Path(a.train_pkl).stem.replace('_train',''),'best_val_auc':max(r['val_auc'] for r in rows),'last_val_loss':rows[-1]['val_loss'],'epochs':a.epochs,'align_w':a.align_w,'domain_w':a.domain_w,'orth_w':a.orth_w})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train_pkl',required=True); p.add_argument('--val_pkl',required=True); p.add_argument('--img_dir',required=True)
    p.add_argument('--out_dir',required=True); p.add_argument('--summary_csv',required=True)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch',type=int,default=12); p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--miss_p',type=float,default=0.3); p.add_argument('--shared_d',type=int,default=128); p.add_argument('--spec_d',type=int,default=128)
    p.add_argument('--align_w',type=float,default=0.15); p.add_argument('--domain_w',type=float,default=0.05); p.add_argument('--orth_w',type=float,default=0.02)
    p.add_argument('--num_workers',type=int,default=0); p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
