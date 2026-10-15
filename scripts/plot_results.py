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

def history():
    stages = [('v1', 'Baseline', BLUE), ('v2', 'Visual generalization', TEAL), ('v3', 'Low-LR refinement', PURPLE)]
    loaded = {}
    for stage, title, color in stages:
        file = f'{stage}_tensorboard.csv' if stage != 'v3' else 'v3_wandb.csv'
        rows = list(csv.DictReader((ROOT / 'results/training' / file).open()))
        loaded[stage] = (np.array([int(float(r['step'])) for r in rows]), rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.1))
    fig.subplots_adjust(top=.82, bottom=.14, hspace=.46, wspace=.28)
    fig.text(.07, .96, 'One project, three training stages', fontsize=23, weight='bold')
    fig.text(.07, .905, 'v0 pipeline validation  →  v1 baseline  →  v2 c2r augmentation  →  v3 refinement + confirmation', fontsize=10.5, color='#526078')
    for col, (stage, title, color) in enumerate(stages):
        steps, rows = loaded[stage]
        for row, (field, ylabel) in enumerate([('training_vla_loss', 'Action loss (log scale)'), ('training_lr', 'Learning rate')]):
            ax = axes[row, col]; y = np.array([float(r[field]) for r in rows])
            ax.plot(steps, y, color=color, alpha=.15, lw=.45)
            ax.plot(steps, smooth(y), color=color, lw=1.8)
            ax.set_xlabel(f'{stage} optimizer step'); ax.set_ylabel(ylabel); ax.grid(axis='y'); ax.set_xlim(1, steps[-1]); ax.xaxis.set_major_locator(MaxNLocator(5))
            if row == 0:
                ax.set_yscale('log'); ax.set_title(f'{stage.upper()} · {title}', loc='left', pad=13)
            else:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.text(.07, .065, 'Independent step axes. v1 and v2 each start from the same LingBot-VLA 2.0 base; only v3 starts from v2@19k.', fontsize=9, color='#526078')
    fig.text(.07, .028, 'Sources: original TensorBoard events (v1/v2), W&B export (v3). Raw trace + 75-record trailing mean; no imputation.', fontsize=9, color='#68758b')
    fig.savefig(OUT / 'training_evolution.png', dpi=170); plt.close(fig)
    for stage, title, color in stages[:2]:
        steps, rows = loaded[stage]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.1))
        fig.subplots_adjust(top=.83, bottom=.12, hspace=.43, wspace=.28)
        fig.text(.07, .955, f'LingBot-VLA 2.0  /  {stage} {title.lower()}', fontsize=23, weight='bold')
        fig.text(.07, .902, f'{steps[-1]:,} recorded steps  |  global batch 1,024  |  Muon  |  original training events', fontsize=11, color='#526078')
        specs = [('Action loss', 'training_vla_loss'), ('Total loss', 'training_loss'), ('Current-depth loss', 'training_depth_loss'), ('Future-video loss', 'training_future_video_loss'), ('Learning rate', 'training_lr'), ('Gradient norm', 'training_grad_norm')]
        for ax, (name, key) in zip(axes.flat, specs):
            y = np.array([float(r[key]) for r in rows]); ax.plot(steps, y, color=color, alpha=.16, lw=.45); ax.plot(steps, smooth(y), color=color, lw=1.8)
            ax.set_title(name, loc='left', pad=13); ax.set_xlabel('Optimizer step'); ax.grid(axis='y'); ax.set_xlim(1, steps[-1]); ax.xaxis.set_major_locator(MaxNLocator(5))
            if 'loss' in key: ax.set_yscale('log'); ax.set_ylabel('Log scale')
            if key == 'training_lr': ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        fig.text(.07, .04, 'Recovered from TensorBoard. Restart duplicates retain the latest observed value per step and metric; no imputation.', fontsize=9, color='#68758b')
        fig.savefig(OUT / f'training_{stage}.png', dpi=170); plt.close(fig)

def v4_training():
    rows=list(csv.DictReader((ROOT/'results/training/v4_tensorboard.csv').open()))
    steps=np.array([int(r['step']) for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    fig.subplots_adjust(left=.065,right=.975,top=.73,bottom=.20,wspace=.3)
    fig.text(.065,.94,'v4  /  balanced continuation from v3@4k',fontsize=23,weight='bold')
    fig.text(.065,.845,'6,000 additional steps · global batch 1,024 · 50% frame + 50% episode sampling',fontsize=11,color='#526078')
    for ax,(key,title,color) in zip(axes,[('training_vla_loss','Action loss',TEAL),('training_lr','Learning rate',BLUE),('training_grad_norm','Gradient norm',PURPLE)]):
        values=np.array([float(r[key]) for r in rows])
        ax.plot(steps,values,color=color,alpha=.2,lw=.5)
        ax.plot(steps,smooth(values),color=color,lw=1.8)
        ax.set_title(title,loc='left');ax.set_xlabel('v4 optimizer step');ax.grid(axis='y');ax.set_xlim(1,6000)
        if key=='training_lr':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.text(.065,.065,'Source: original TensorBoard events · 6,000 records · raw trace + 75-record mean · evaluation gains are not established',fontsize=9,color='#68758b')
    fig.savefig(OUT/'training_v4.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    training();scores();history();v4_training();print(OUT)
