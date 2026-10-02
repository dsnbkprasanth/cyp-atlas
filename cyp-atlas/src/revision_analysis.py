"""Referee-revision analyses: CI/PR-AUC/MCC, model-comparison significance,
calibration, applicability-domain threshold + reliability, domain-split
prevalence, HDI validation composition + constituent overlap.

Run from cyp-atlas/src AFTER: train_cyp.py, atlas.py, domain.py, risk.py.

    python revision_analysis.py

Outputs -> results/revision/ (CSV tables) and results/revision/figures/ (PNG).
ponytail: reliability curve computed for CYP3A4 (largest set); rerun per isoform
by looping TASKS if a reviewer wants all five.
"""
import csv, os, json
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LABELS="data/cyp_labels.csv"; ATLAS="data/cyp_atlas_domain.csv"
RISK="results/risk_validation.csv"; SEED="../../hdi-gnn/src/data/hdi_seed.csv"
OUT="results/revision"; FIG=OUT+"/figures"; os.makedirs(FIG,exist_ok=True)
TASKS=["CYP1A2","CYP2C9","CYP2C19","CYP2D6","CYP3A4"]
rng=np.random.default_rng(0)

def fp(smi,n=1024,r=2):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    a=np.zeros((n,),dtype=np.int8)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=n),a); return a
def bv(smi,n=2048,r=2):
    m=Chem.MolFromSmiles(smi); return AllChem.GetMorganFingerprintAsBitVect(m,r,nBits=n) if m else None

rows=list(csv.DictReader(open(LABELS)))
X=[];keep=[]
for i,r in enumerate(rows):
    f=fp(r["smiles"])
    if f is not None: X.append(f);keep.append(i)
X=np.array(X); rows=[rows[i] for i in keep]
RF=lambda:RandomForestClassifier(n_estimators=300,random_state=0,n_jobs=-1)
LR=lambda:LogisticRegression(max_iter=1000); HG=lambda:HistGradientBoostingClassifier(random_state=0)
def oofp(Xf,yf,make):
    p=np.zeros(len(yf))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,yf):
        p[te]=make().fit(Xf[tr],yf[tr]).predict_proba(Xf[te])[:,1]
    return p
def ci(y,p,n=2000):
    idx=np.arange(len(y));v=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(set(y[b]))>1: v.append(roc_auc_score(y[b],p[b]))
    return np.percentile(v,2.5),np.percentile(v,97.5)

# 1. full metrics + store OOF
full=[];store={}
for t in TASKS:
    m=np.array([r[t] in("0","1") for r in rows]); Xf=X[m]
    yf=np.array([int(r[t]) for r in rows if r[t] in("0","1")])
    p=oofp(Xf,yf,RF); store[t]=(Xf,yf,p)
    lo,hi=ci(yf,p); pred=(p>=0.5).astype(int)
    sens=((pred==1)&(yf==1)).sum()/max((yf==1).sum(),1)
    spec=((pred==0)&(yf==0)).sum()/max((yf==0).sum(),1)
    full.append([t,len(yf),int(yf.sum()),round(roc_auc_score(yf,p),3),f"{lo:.3f}-{hi:.3f}",
                 round(average_precision_score(yf,p),3),round(matthews_corrcoef(yf,pred),3),
                 round(sens,3),round(spec,3),round(brier_score_loss(yf,p),3)])
csv.writer(open(OUT+"/metrics_full.csv","w",newline="")).writerows(
  [["enzyme","n","positives","AUROC","AUROC_95CI","PR_AUC","MCC","sensitivity","specificity","Brier"]]+full)

# 2. model comparison + paired bootstrap
comp=[]
for t in TASKS:
    Xf,yf,pr=store[t]; plr=oofp(Xf,yf,LR); phg=oofp(Xf,yf,HG)
    row=[t,round(roc_auc_score(yf,pr),3),round(roc_auc_score(yf,plr),3),round(roc_auc_score(yf,phg),3)]
    for pb in (plr,phg):
        idx=np.arange(len(yf));df=[]
        for _ in range(2000):
            b=rng.choice(idx,len(idx),replace=True)
            if len(set(yf[b]))>1: df.append(roc_auc_score(yf[b],pr[b])-roc_auc_score(yf[b],pb[b]))
        df=np.array(df); row+=[f"{df.mean():.3f}",f"{(df<=0).mean():.4f}"]
    comp.append(row)
csv.writer(open(OUT+"/model_compare_sig.csv","w",newline="")).writerows(
  [["enzyme","RF","LR","HG","dRF-LR","p(RF>LR)","dRF-HG","p(RF>HG)"]]+comp)

# 3. calibration
plt.figure(figsize=(6,5))
for t in TASKS:
    _,yf,p=store[t]; fr,mn=calibration_curve(yf,p,n_bins=10,strategy="quantile")
    plt.plot(mn,fr,marker="o",lw=1,label=t)
plt.plot([0,1],[0,1],"k--",lw=.8); plt.xlabel("Mean predicted probability")
plt.ylabel("Observed inhibitor fraction"); plt.title("Calibration"); plt.legend(fontsize=8)
plt.savefig(FIG+"/fig6_calibration.png",dpi=200,bbox_inches="tight"); plt.close()

# 4. AD reliability (CYP3A4): NN Tanimoto to training vs AUROC
t="CYP3A4"; Xf,yf,p=store[t]
sub=[r["smiles"] for r in rows if r[t] in("0","1")]; bvs=[bv(s) for s in sub]
samp=rng.choice(len(sub),min(2500,len(sub)),replace=False)
pool=[bvs[i] for i in rng.choice(len(sub),min(4000,len(sub)),replace=False)]
nn=np.zeros(len(samp))
for k,i in enumerate(samp):
    if bvs[i] is None: continue
    s=DataStructs.BulkTanimotoSimilarity(bvs[i],[b for b in pool if b is not None]); nn[k]=max(s) if s else 0
ys=yf[samp]; ps=p[samp]; rel=[]
for a,b in [(0,.2),(.2,.3),(.3,.4),(.4,.6),(.6,1.01)]:
    mm=(nn>=a)&(nn<b)
    if mm.sum()>20 and len(set(ys[mm]))>1:
        rel.append([f"{a:.2f}-{b:.2f}",int(mm.sum()),round(roc_auc_score(ys[mm],ps[mm]),3),round(brier_score_loss(ys[mm],ps[mm]),3)])
csv.writer(open(OUT+"/ad_reliability.csv","w",newline="")).writerows([["NN_Tanimoto_bin","n","AUROC","Brier"]]+rel)
plt.figure(figsize=(6,4)); plt.bar([r[0] for r in rel],[r[2] for r in rel]); plt.ylim(0,1)
plt.axhline(0.5,color="k",ls="--",lw=.8); plt.xlabel("NN Tanimoto to training"); plt.ylabel("AUROC (CYP3A4)")
plt.title("Reliability vs training similarity"); plt.savefig(FIG+"/fig7_ad_reliability.png",dpi=200,bbox_inches="tight"); plt.close()

# 5. AD threshold + domain-split prevalence
adom=list(csv.DictReader(open(ATLAS))); sim=np.array([float(r["max_train_tanimoto"]) for r in adom])
csv.writer(open(OUT+"/ad_threshold.csv","w",newline="")).writerows(
  [["Tanimoto_threshold","in_domain_fraction","in_domain_n"]]+[[t0,round((sim>=t0).mean(),3),int((sim>=t0).sum())] for t0 in (.2,.25,.3,.4,.5)])
inm=sim>=0.3; dsp=[]
cols=[c for c in adom[0] if c in TASKS]
for t in cols:
    pr=np.array([float(r[t]) for r in adom])
    dsp.append([t,round((pr[inm]>=.5).mean(),3),round((pr[~inm]>=.5).mean(),3),round((pr>=.5).mean(),3)])
csv.writer(open(OUT+"/domain_split_prevalence.csv","w",newline="")).writerows(
  [["isoform","in_domain_prev","out_domain_prev","overall_prev"]]+dsp)

# 6. HDI composition + overlap
rv=list(csv.DictReader(open(RISK))); seed=list(csv.DictReader(open(SEED)))
csm={r["constituent"]:r["smiles_a"] for r in seed}
trbv=[b for b in (bv(r["smiles"]) for r in rows) if b is not None]
ov=[]
for name,smi in csm.items():
    b=bv(smi); mx=max(DataStructs.BulkTanimotoSimilarity(b,trbv)) if b else 0
    ov.append([name,round(mx,3),"exact" if mx>=.999 else("near" if mx>=.4 else "distant")])
csv.writer(open(OUT+"/hdi_overlap.csv","w",newline="")).writerows([["constituent","max_train_tanimoto","category"]]+ov)
summ={"positive_pairs":sum(r["label"]=="1" for r in rv),"negative_pairs":sum(r["label"]=="0" for r in rv),
      "unique_constituents":len({r["herb"] for r in rv}),"unique_drugs":len({r["drug"] for r in rv}),
      "curated_positive_total":len(seed),
      "exact":sum(1 for o in ov if o[2]=="exact"),"near":sum(1 for o in ov if o[2]=="near"),
      "distant":sum(1 for o in ov if o[2]=="distant")}
json.dump(summ,open(OUT+"/summary.json","w"),indent=2)
assert full and rel
print("wrote results/revision/  ->",summ)
