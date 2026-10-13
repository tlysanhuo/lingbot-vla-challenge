"""Rebuild repository figures from committed W&B exports and episode counts."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelcolor':'#526078','text.color':'#152238','axes.titleweight':'bold',
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#d8dfea',
 'grid.color':'#e5eaf2','grid.alpha':0.8,'figure.facecolor':'#f7f9fc','axes.facecolor':'white',
 'savefig.facecolor':'#f7f9fc'})
TEAL, BLUE, PURPLE = '#008b83', '#4263eb', '#9061c2'
def smooth(y, window=75):
    return np.array([np.mean(y[max(0,i-window+1):i+1]) for i in range(len(y))])

def training():
    rows=list(csv.DictReader((ROOT/'results/training/v3_wandb.csv').open()))
    steps=np.array([int(float(r['step'])) for r in rows])
    y=lambda key: np.array([float(r[key]) for r in rows])
    fig,axes=plt.subplots(2,3,figsize=(15,8.2))
    fig.subplots_adjust(top=.84,bottom=.12,hspace=.43,wspace=.28)
    fig.text(.07,.955,'LingBot-VLA 2.0  /  v3 post-training',fontsize=23,weight='bold')
    fig.text(.07,.905,'4,000 optimizer steps  |  global batch 1,024  |  Muon  |  cosine 2e-5 → 2e-6',fontsize=11,color='#526078')
    specs=[('Action loss','training_vla_loss',TEAL),('Total training loss','training_loss',BLUE),
           ('Geometry alignment','training_depth_loss',PURPLE),('Future-video loss','training_future_video_loss',TEAL),
           ('Learning rate','training_lr',BLUE),('Gradient norm','training_grad_norm',PURPLE)]
    for ax,(title,key,color) in zip(axes.flat,specs):
        data=y(key)
        ax.plot(steps,data,color=color,alpha=.18,lw=.55)
        ax.plot(steps,smooth(data),color=color,lw=1.9,label='75-record trailing mean')
        if key=='training_depth_loss':
            ax.plot(steps,smooth(y('training_future_depth_loss')),color=BLUE,lw=1.5,label='Future depth')
            ax.legend(frameon=False,fontsize=8)
        ax.set_title(title,loc='left',pad=13);ax.set_xlabel('Optimizer step');ax.grid(axis='y');ax.set_xlim(1,4000);ax.xaxis.set_major_locator(MaxNLocator(5))
        if key=='training_lr':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.text(.07,.045,f'Source: W&B export · {len(rows):,} records · raw trace + trailing mean · missing records are not imputed',fontsize=9,color='#68758b')
    fig.savefig(OUT/'training_v3.png',dpi=170);fig.savefig(OUT/'training_v3.svg');plt.close(fig)

def scores():
    data=json.loads((ROOT/'results/evaluation/v3_confirmation.json').read_text())['results']
    fig,ax=plt.subplots(figsize=(10.8,4.9));fig.subplots_adjust(left=.09,right=.96,top=.77,bottom=.19)
    vals=[100*sum(x['successes'] for x in data[s].values())/sum(x['attempts'] for x in data[s].values()) for s in ['clean','randomized']]
    vals.append(sum(vals)/2)
    fig.text(.09,.92,'50 tasks. One policy. 1,000 episodes.',fontsize=23,weight='bold')
    fig.text(.09,.845,'v3 4k raw  ·  FP32 inference  ·  chunk 50 / execute 30  ·  10 denoising steps',color='#526078',fontsize=10.5)
    bars=ax.barh(['Clean / seen','Randomized / unseen','Equal-setting mean'],vals,color=[TEAL,BLUE,PURPLE],height=.55)
    for bar,v in zip(bars,vals):ax.text(v+1.2,bar.get_y()+bar.get_height()/2,f'{v:.1f}%',va='center',fontsize=15,weight='bold')
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Success rate (%)');ax.set_axisbelow(True);ax.grid(axis='x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.text(.09,.055,'Internal confirmation: 404/500 clean + 252/500 randomized. Not an official competition ranking.',color='#68758b',fontsize=9)
    fig.savefig(OUT/'confirmation.png',dpi=170);plt.close(fig)
    tasks=sorted(data['clean']);values=np.array([[data[s][task]['successes'] for s in ['clean','randomized']] for task in tasks])
    fig,axes=plt.subplots(1,2,figsize=(12,10));fig.subplots_adjust(left=.25,right=.96,top=.89,bottom=.06,wspace=1.25)
    fig.suptitle('Task-level confirmation  /  successes out of 10',x=.52,y=.975,fontsize=19,weight='bold')
    for ax,start in zip(axes,[0,25]):
        arr=values[start:start+25];ax.imshow(arr,cmap='YlGnBu',vmin=0,vmax=10,aspect='auto')
        ax.set_yticks(range(25),[x.replace('_',' ') for x in tasks[start:start+25]],fontsize=8)
        ax.set_xticks([0,1],['Clean / seen','Random / unseen'],fontsize=9);ax.xaxis.tick_top();ax.tick_params(length=0,pad=7)
        for i in range(25):
            for j in range(2):ax.text(j,i,str(arr[i,j]),ha='center',va='center',fontsize=9,color='white' if arr[i,j]>=7 else '#152238')
    fig.savefig(OUT/'task_breakdown.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    training();scores();print(OUT)
