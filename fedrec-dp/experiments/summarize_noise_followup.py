"""Verify and tabulate completed E3/E4 evidence without training or scoring."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
E3 = ROOT/'results/extensions/noise_replication_v1'
E4 = ROOT/'results/extensions/ranking_tails_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path):
    freeze=json.loads(path.read_text())
    for name,expected in freeze['sources'].items():
        assert sha(ROOT/name)==expected, name
    return len(freeze['sources'])


def verify_summary(root, data, keys, drop):
    columns=[c for c in data.select_dtypes(include='number') if c not in keys+drop]
    calculated=data.groupby(keys)[columns].mean().sort_index()
    saved=pd.read_csv(root/'by_seed.csv').set_index(keys).sort_index()
    # E4 retains a deliberately smaller whitelist of numerical columns.
    columns=[c for c in columns if c in saved]
    assert calculated.index.equals(saved.index)
    assert np.allclose(calculated[columns],saved[columns],rtol=2e-8,atol=1e-10)
    stats=['mean','median','std','min','max'] if root==E3 else ['mean','std','min','max']
    calculated=saved.groupby(keys[:-1])[columns].agg(stats)
    calculated.columns=['_'.join(c) for c in calculated.columns]
    summary=pd.read_csv(root/'summary.csv').set_index(keys[:-1]).sort_index()
    assert calculated.index.equals(summary.index)
    assert np.allclose(calculated,summary[calculated.columns],rtol=2e-8,atol=1e-10)


def main():
    audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),training_executed=False,
               model_scores_computed=False,summary_script_sha256=sha(__file__))
    audit['frozen_source_hashes']={'E3':frozen(E3/'freeze.json'),'E4':frozen(E4/'freeze.json')}
    data=json.loads((E3/'dataset.json').read_text())
    assert data==json.loads((ROOT/'data/extensions/ml-1m_noise_v1/dataset.json').read_text())
    for name,expected in {**data['raw_sha256'],**data['split_sha256']}.items():
        assert sha(ROOT/'data/extensions/ml-1m_noise_v1'/name)==expected
    assert sha(ROOT/'data/extensions/ml-1m_noise_v1/dataset.json')==json.loads((E3/'freeze.json').read_text())['dataset_sha256']
    training=json.loads((E3/'training_records.json').read_text())
    assert len(training)==30 and all(r['status']=='ok' and r['round']==50 for r in training)
    validation=[]
    for r in training:
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        ranks=pd.read_csv(E3/'training'/f"{r['tag']}_validation.csv")
        assert len(ranks)==6035 and ranks.user.is_unique and ranks['rank'].between(1,3706).all()
        ndcg=np.where(ranks['rank']<=10,1/np.log2(ranks['rank']+1),0).mean()
        assert np.isclose(ndcg,r['validation']['ndcg@10'],rtol=0,atol=1e-14)
        validation.append(dict(unit=r['unit'],level=r['level'],seed=r['seed'],validation_ndcg=ndcg,
                               max_user_norm=r['final_user_norm_max'],total_bytes=r['total_bytes']))
    val=pd.concat([pd.DataFrame(validation),pd.read_csv(E3/'popularity_validation.csv')],ignore_index=True)
    val.to_csv(E3/'validation_by_seed.csv',index=False,float_format='%.12g')
    val.groupby(['unit','level']).validation_ndcg.agg(['mean','std','min','max']).to_csv(
        E3/'validation_summary.csv',float_format='%.12g')
    observations=[]
    maximum_alignment=0.
    for path in (E3/'probes').glob('*.json'):
        r=json.loads(path.read_text())
        assert r['status']=='ok' and len(r['observations'])==14
        assert len(r['observer_users'])==128
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        for row in r['observations']:
            assert row['users_exposed']+row['users_unexposed']==128
            assert 0<=row['top10_churn']<=1
            maximum_alignment=max(maximum_alignment,row['alignment_product_error_max'])
            if row['phase']=='after_reset' and row['lag']==0:
                if row['branch']=='shared_reset':
                    assert row['q_difference_fro']==0
                if row['branch']=='local_reset':
                    assert row['p_difference_fro']==0
            observations.append({**{k:r[k] for k in ['dataset','unit','level','balancing','coupling','seed','replicate']},**row})
    assert len(observations)==2800
    obs=pd.DataFrame(observations)
    keys=['dataset','unit','level','balancing','coupling','phase','lag','branch','seed']
    assert obs.groupby(keys).size().eq(2).all()
    verify_summary(E3,obs,keys,['replicate'])
    seed=pd.read_csv(E3/'by_seed.csv')
    endpoint=seed[(seed.phase=='after_reset')&(seed.lag==10)&(seed.branch=='shared_reset')]
    endpoint.to_csv(E3/'endpoint_by_seed.csv',index=False,float_format='%.12g')
    d=endpoint[endpoint.unit=='two_r8']
    wide=d.pivot(index=['dataset','level','balancing','seed'],columns='coupling',values='top10_churn')
    wide['raw_minus_aligned']=wide.raw-wide.aligned
    wide.to_csv(E3/'coupling_contrasts_by_seed.csv',float_format='%.12g')
    summary=endpoint.groupby(['dataset','unit','level','balancing','coupling'])[
        ['top10_churn','score_rms','score_ratio_to_pulse','validation_ndcg_difference']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(E3/'endpoint_summary.csv',float_format='%.12g')
    audit['e3']=dict(training_runs=30,probes=200,observations=2800,failures=0,
                     validation_ranks_recomputed=30*6035,alignment_product_relative_error_max=maximum_alignment,
                     all_synthetic_checks_passed=all(r['passed'] for r in json.loads((E3/'synthetic_checks.json').read_text())['records']))
    pairs=[]
    for path in (E4/'states').glob('*.json'):
        r=json.loads(path.read_text())
        assert r['status']=='ok' and len(r['observations'])==64 and r['flagged_pairs']==0
        assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
        pairs.extend({**{k:r[k] for k in ['dataset','unit','level','seed']},**row} for row in r['observations'])
    f=pd.DataFrame(pairs)
    assert len(f)==3200 and f.status.eq('ok').all()
    assert np.allclose(f.risk_gaussian_matched,ndtr(-f.margin/np.sqrt(f.variance)),rtol=1e-12,atol=1e-15)
    assert np.allclose(f.variance,f.linear_variance+2*f['rank']*f.tau**4,rtol=1e-12)
    assert np.allclose(f.absolute_error_matched,abs(f.risk_gaussian_matched-f.exact_risk_numerical),rtol=1e-12)
    verify_summary(E4,f,['dataset','unit','level','pair','seed'],['user','item_high','item_low','rank'])
    checks=json.loads((E4/'synthetic_checks.json').read_text())
    assert checks['all_passed'] and len(checks['records'])==6
    primary=f[f.pair.isin(['10_11','1_11'])]
    fractions=primary.groupby('dataset').error_matched_over_01.mean()
    assert fractions.eq(0).all()
    summary=f.groupby(['dataset','unit','level'])[
        ['absolute_error_matched','absolute_error_linear','bilinear_share','exact_risk_numerical']].agg(['mean','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(E4/'screen_summary.csv',float_format='%.12g')
    audit['e4']=dict(states=50,pairs=3200,flagged_pairs=0,synthetic_cases_passed=6,
                     primary_pairs=len(primary),primary_fraction_error_over_01=fractions.to_dict(),
                     max_absolute_error_matched=float(f.absolute_error_matched.max()),
                     max_absolute_error_linear=float(f.absolute_error_linear.max()),
                     max_quadrature_error_estimate=float(f.quadrature_error_estimate.max()),
                     max_omitted_tail_mass=float(f.omitted_tail_mass.max()),advancement_gate_passed=False)
    audit['all_checks_passed']=True
    for root in [E3,E4]:
        (root/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
