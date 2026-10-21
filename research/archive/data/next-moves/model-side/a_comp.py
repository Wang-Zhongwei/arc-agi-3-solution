from load import *
import numpy as np, sys
runs = UC10 if len(sys.argv) < 2 else sys.argv[1:]
R=[]
for run in runs: R+=load(run)[1]
cats=['sys_chars','user_text','tool_chars','asst_reason','asst_code','asst_content']
X=np.array([[r[c] for c in cats]+[r['n_msgs'],1.0] for r in R],dtype=float)
y=np.array([r['pt']-(r['img'] or 0) for r in R],dtype=float)
coef,res,rk,sv=np.linalg.lstsq(X,y,rcond=None)
pred=X@coef
print('tokens per char by category:', {c:round(v,3) for c,v in zip(cats+['per_msg','const'],coef)})
print('R2', round(1-((y-pred)**2).sum()/((y-y.mean())**2).sum(),4))
pt=np.array([r['pt'] for r in R],float); img=np.array([r['img'] or 0 for r in R],float)
print(f"requests {len(R)}; prompt tokens mean {pt.mean():.0f} median {np.median(pt):.0f}; images in context median {np.median([r['n_img'] for r in R]):.0f}; image tokens share of prompt (token-weighted) {img.sum()/pt.sum()*100:.1f}%")
shares={c: (X[:,i]*coef[i]).sum()/pt.sum()*100 for i,c in enumerate(cats)}
shares['per-message overhead+const']=((X[:,6]*coef[6]).sum()+coef[7]*len(R))/pt.sum()*100
shares['images']=img.sum()/pt.sum()*100
print('token-weighted prompt composition %:', {k:round(v,1) for k,v in shares.items()})
cached=np.array([r['cached'] or 0 for r in R],float)
print(f"cached share {cached.sum()/pt.sum()*100:.1f}%; uncached tokens per request median {np.median(pt-cached):.0f}; total prefill {((pt-cached).sum())/1e6:.1f}M vs generated {sum(r['ct'] for r in R)/1e6:.1f}M")
