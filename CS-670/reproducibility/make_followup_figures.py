"""E3/E4 report figures from completed compact evidence; no evaluation."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/cs670-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=220)
    plt.close(fig)


def main():
    d=pd.read_csv(ROOT/'results/noise_replication/endpoint_by_seed.csv')
    d=d[d.unit=='two_r8']
    modes=[('svd','raw'),('svd','aligned'),('none','raw'),('none','aligned')]
    labels=['SVD\nraw','SVD\naligned','No balance\nraw','No balance\naligned']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,dataset in enumerate(['ml100k','ml1m']):
        for col,level in enumerate(['eps1','eps2']):
            ax=axes[row,col]
            for x,(balance,coupling) in enumerate(modes):
                values=100*d[(d.dataset==dataset)&(d.level==level)&(d.balancing==balance)&(d.coupling==coupling)].sort_values('seed').top10_churn.to_numpy()
                assert len(values)==5
                mean,sd=values.mean(),values.std(ddof=1)
                color='#b35806' if coupling=='raw' else '#2166ac'
                ax.errorbar([x],[mean],yerr=np.array([[min(mean,sd)],[sd]]),fmt='D',capsize=4,color=color)
                ax.scatter(x+np.linspace(-.1,.1,5),values,facecolors='none',edgecolors=color,s=30)
            ax.set_yscale('symlog',linthresh=.02)
            ax.set_ylim(0,40)
            ax.set_yticks([0,.01,.1,1,10,30],labels=['0','.01','.1','1','10','30'])
            ax.set_xticks(range(4),labels)
            name={'ml100k':'MovieLens-100K','ml1m':'MovieLens-1M'}[dataset]
            ax.set(title=f'{name}, starting epsilon {level[3:]}',ylabel='Top-10 set disagreement (%)')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('E3: large coupling differences depend on factor rebalancing\nShared reset, ten rounds later; five seeds and two draws per seed',fontsize=12)
    save(fig,'07_balancing_replication')

    d=pd.read_csv(ROOT/'results/noise_replication/validation_by_seed.csv')
    methods=[('full','Full','#444444'),('two_r8','Two r8','#2166ac'),
             ('fixed_r8','FixedB r8','#b35806'),('dp_popularity','DP popularity','#1b7837')]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
    for ax,level in zip(axes,['eps1','eps2']):
        for x,(unit,label,color) in enumerate(methods):
            values=d[(d.unit==unit)&(d.level==level)].sort_values('seed').validation_ndcg.to_numpy()
            ax.bar(x,values.mean(),yerr=values.std(ddof=1),color=color,alpha=.8,capsize=4)
            ax.scatter(x+np.linspace(-.1,.1,5),values,s=18,c='black',zorder=4)
        ax.set_xticks(range(4),[m[1] for m in methods],rotation=15)
        ax.set(title=f'Target epsilon {level[3:]}',ylabel='Validation NDCG@10',ylim=(0,.035))
    fig.suptitle('MovieLens-1M: frozen setting transfer, 50 collaborative rounds\nFull-population validation; test targets not scored',fontsize=12)
    save(fig,'08_ml1m_validation')

    d=pd.read_csv(ROOT/'results/ranking_tails/screen_summary.csv')
    groups=[('ml100k',f'two_r{r}',f'100K\nr{r}') for r in [4,8,16,32]]+[('ml1m','two_r8','1M\nr8')]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained',sharey=True)
    for ax,level in zip(axes,['eps1','eps2']):
        for offset,column,label,color in [(-.16,'absolute_error_matched_max','Variance-matched Gaussian','#2166ac'),
                                           (.16,'absolute_error_linear_max','Linearized Gaussian','#b35806')]:
            values=[d[(d.dataset==dataset)&(d.unit==unit)&(d.level==level)].iloc[0][column] for dataset,unit,_ in groups]
            ax.bar(np.arange(5)+offset,values,width=.3,color=color,label=label)
        ax.axhline(.01,color='#555555',linestyle='--',label='Per-pair advancement threshold')
        ax.set_yscale('log')
        ax.set_ylim(1e-7,.025)
        ax.set_xticks(range(5),[x[2] for x in groups])
        ax.set(title=f'Starting epsilon {level[3:]}',ylabel='Maximum absolute flip-probability error')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.2))
    fig.suptitle('E4: the ranking-tail hypothesis failed the declared screen\nEach maximum includes five states × sixteen users × four fixed rank pairs',fontsize=12)
    save(fig,'09_ranking_tail_screen')


if __name__=='__main__':
    main()
