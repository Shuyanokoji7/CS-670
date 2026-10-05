"""E4: fixed-state conditional ranking-risk audit; no labels or training."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.threads import force_env
force_env()
import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.integrate import quad, IntegrationWarning
from scipy.special import ndtr
from scipy.stats import ncx2
from src.threads import enforce_and_report

OUT = ROOT / 'results/extensions/ranking_tails_v1'
PROTOCOL = ROOT / 'docs/extensions/RANKING_TAIL_PROTOCOL.md'
E1 = ROOT / 'results/extensions/effective_noise_v1'
E3 = ROOT / 'results/extensions/noise_replication_v1'
PAIRS = [(10,11), (1,11), (10,100), (1,100)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def probability(a, b):
    """Known product-normal law via noncentral-chi-square difference."""
    rank = len(a)
    lp, lm = float(np.sum((a+b)**2)/2), float(np.sum((a-b)**2)/2)
    tail = 1e-12
    lower = float(np.sqrt(ncx2.ppf(tail, rank, lm)))
    upper = float(np.sqrt(ncx2.isf(tail, rank, lm)))
    omitted = float(ncx2.cdf(lower**2, rank, lm) + ncx2.sf(upper**2, rank, lm))
    assert np.isfinite([lp, lm, lower, upper, omitted]).all()
    # The square-root transform removes the r=1 density singularity at zero.
    def integrand(y):
        if y <= 0:
            return 0.
        return float(2*y*np.exp(ncx2.logpdf(y*y, rank, lm))*ncx2.cdf(y*y, rank, lp))
    center = float(np.sqrt(rank+lp))
    points = [x for x in [center-2, center, center+2] if lower < x < upper]
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(integrand, lower, upper, epsabs=2e-10, epsrel=1e-8,
                            limit=120, points=points)
    assert np.isfinite([value, error]).all() and -.00000001 <= value <= 1.00000001
    return dict(exact_risk_numerical=float(value), quadrature_error_estimate=float(error),
                omitted_tail_mass=omitted, lambda_plus=lp, lambda_minus=lm)


def synthetic():
    path = OUT / 'synthetic_checks.json'
    if path.exists():
        raise FileExistsError(path)
    enforce_and_report()
    patterns = [(np.array([x]), np.array([x])) for x in [.5, 1., 4.]]
    patterns += [(np.linspace(.2,1.4,r), np.linspace(1.2,.8,r)) for r in [4,8]]
    patterns += [(np.zeros(8), np.zeros(8))]
    rows = []
    for i, (a,b) in enumerate(patterns):
        rank, mean = len(a), float(a @ b)
        variance = float(a @ a + b @ b + rank)
        third = 6*mean
        result = probability(a,b)
        rng = np.random.default_rng([6721,i])
        n = 200000
        values = ((a+rng.normal(size=(n,rank)))*(b+rng.normal(size=(n,rank)))).sum(axis=1)
        central = values-mean
        estimates = [values, central**2, central**3]
        targets = [mean, variance, third]
        zs = [(float(v.mean())-target)/(float(v.std(ddof=1))/np.sqrt(n))
              for v,target in zip(estimates,targets)]
        assert max(abs(x) for x in zs) < 6
        empirical = float(np.mean(values <= 0))
        risk = result['exact_risk_numerical']
        se = np.sqrt(risk*(1-risk)/n)
        assert abs(empirical-risk) < 6*se+1/n
        if rank == 1:
            closed = float(ndtr(-a[0])*ndtr(b[0])+ndtr(a[0])*ndtr(-b[0]))
            assert abs(risk-closed) < 1e-9
            result['rank_one_closed_form'] = closed
        if mean == 0:
            assert abs(risk-.5) < 1e-9
        rotation, _ = np.linalg.qr(rng.normal(size=(rank,rank)))
        rotated = probability(rotation@a, rotation@b)['exact_risk_numerical']
        assert abs(rotated-risk) < 1e-9
        # p -> c*p makes both Bp and sy scale by c, preserving b and the law.
        u, sy, factor = b*2.7, 2.7, 13.
        assert np.allclose((factor*u)/(factor*sy), b, rtol=1e-14, atol=1e-14)
        rows.append(dict(case=i, rank=rank, a=a.tolist(), b=b.tolist(), passed=True,
                         mean=mean, variance=variance, third_cumulant=third,
                         gaussian_variance_matched=float(ndtr(-mean/np.sqrt(variance))),
                         bilinear_share=rank/variance, monte_carlo_draws=n,
                         empirical_risk=empirical, risk_mc_se=float(se), moment_z_scores=zs,
                         **result))
    write(path, dict(created_utc=stamp(), code_sha256=sha(__file__), protocol_sha256=sha(PROTOCOL),
                     records=rows, all_passed=True))
    print(json.dumps(dict(synthetic_cases=len(rows), all_passed=True)))


def inputs():
    a = [dict(r, dataset='ml100k') for r in json.loads((E1/'final_records.json').read_text())
         if r['unit'].startswith('two_') and r['level'] in ['eps1','eps2']]
    b = [dict(r, dataset='ml1m') for r in json.loads((E3/'training_records.json').read_text())
         if r['unit'] == 'two_r8']
    assert len(a) == 40 and len(b) == 10
    return a+b


def data_directory(dataset):
    return ROOT / ('data/processed/ml-100k' if dataset == 'ml100k' else 'data/extensions/ml-1m_noise_v1/processed')


def real_job(record):
    enforce_and_report()
    start = time.perf_counter()
    dataset = record['dataset']
    tag = f"{dataset}_{record['unit']}_{record['level']}_seed{record['seed']}"
    checkpoint = ROOT / record['checkpoint']
    expected = record['checkpoint_sha256'] if dataset == 'ml1m' else record['artifact_hashes'][record['checkpoint']]
    assert sha(checkpoint) == expected
    result = dict(tag=tag, dataset=dataset, unit=record['unit'], level=record['level'], seed=record['seed'],
                  checkpoint=record['checkpoint'], checkpoint_sha256=expected, created_utc=stamp(),
                  status='ok', observations=[])
    if record.get('status','ok') != 'ok':
        result.update(status='unavailable_starting_state', failure=record.get('failure'))
        write(OUT/'states'/f'{tag}.json', result)
        return dict(tag=tag, status=result['status'])
    directory = data_directory(dataset)
    meta = json.loads((directory/'meta.json').read_text())
    train = pd.read_csv(directory/'train.csv')
    positive = {int(u): g.item.to_numpy(dtype=int) for u,g in train.groupby('user')}
    code = 100 if dataset == 'ml100k' else 1000
    users = np.sort(np.random.default_rng([6720,code]).choice(meta['n_users'],16,replace=False))
    with np.load(checkpoint, allow_pickle=False) as values:
        P,A,B = [values[k].astype(np.float64) for k in ['P','A','B']]
    c = record['settings']
    tau = c['server_lr']*c['sigma']*c['clip_norm']/(c['q']*len(P))
    rank = A.shape[1]
    for user in users:
        norm = float(np.linalg.norm(P[user]))
        if norm == 0:
            result['observations'].extend(dict(user=int(user), pair=f'{i}_{j}', status='zero_user') for i,j in PAIRS)
            continue
        p = P[user]/norm
        u = B@p
        scores = A@u
        scores[positive[int(user)]] = -np.inf
        ordered = np.argsort(-scores, kind='stable')
        for high,low in PAIRS:
            i,j = ordered[[high-1,low-1]]
            v = A[i]-A[j]
            m = float(v@u)
            sx,sy = np.sqrt(2)*tau,tau
            k = sx**2*sy**2
            linear_variance = float(sx**2*(u@u)+sy**2*(v@v))
            variance = linear_variance+rank*k
            row = dict(user=int(user), item_high=int(i), item_low=int(j), pair=f'{high}_{low}',
                       status='ok' if m > 0 else 'tied_or_reversed', rank=rank,
                       tau=tau, original_user_norm=norm, margin=m,
                       linear_variance=linear_variance, variance=variance,
                       standardized_margin=m/np.sqrt(variance), bilinear_share=rank*k/variance,
                       risk_gaussian_linear=float(ndtr(-m/np.sqrt(linear_variance))),
                       risk_gaussian_matched=float(ndtr(-m/np.sqrt(variance))))
            try:
                row.update(probability(v/sx,u/sy))
                row['absolute_error_matched'] = abs(row['risk_gaussian_matched']-row['exact_risk_numerical'])
                row['absolute_error_linear'] = abs(row['risk_gaussian_linear']-row['exact_risk_numerical'])
                row['error_matched_over_01'] = float(row['absolute_error_matched'] > .01)
                if row['exact_risk_numerical'] >= 1e-6:
                    row['relative_error_matched'] = row['absolute_error_matched']/row['exact_risk_numerical']
            except (AssertionError, ValueError, FloatingPointError, IntegrationWarning) as exc:
                row.update(status='numerical_failure', failure=repr(exc))
            result['observations'].append(row)
    result['wall_time_s'] = time.perf_counter()-start
    result['flagged_pairs'] = sum(row['status'] != 'ok' for row in result['observations'])
    write(OUT/'states'/f'{tag}.json', result)
    return {k:result[k] for k in ['tag','status','flagged_pairs','wall_time_s']}


def summarize():
    rows, inventory = [], []
    for path in sorted((OUT/'states').glob('*.json')):
        record = json.loads(path.read_text())
        observations = record.pop('observations')
        inventory.append(record)
        for row in observations:
            rows.append({**{k:record[k] for k in ['dataset','unit','level','seed']},**row})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'pairs.csv', index=False, float_format='%.12g')
    pd.DataFrame(inventory).to_csv(OUT/'inventory.csv', index=False)
    complete = frame[frame.status == 'ok']
    keys = ['dataset','unit','level','pair','seed']
    columns = ['exact_risk_numerical','risk_gaussian_linear','risk_gaussian_matched',
               'absolute_error_matched','absolute_error_linear','bilinear_share','standardized_margin','error_matched_over_01']
    seeds = complete.groupby(keys)[columns].mean().reset_index()
    seeds.to_csv(OUT/'by_seed.csv', index=False, float_format='%.12g')
    summary = seeds.groupby(keys[:-1])[columns].agg(['mean','std','min','max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT/'summary.csv', float_format='%.12g')
    primary = complete[complete.pair.isin(['10_11','1_11'])]
    gate = primary.groupby('dataset').error_matched_over_01.mean()
    status = dict(created_utc=stamp(), states=len(inventory), pairs=len(frame),
                  flagged_pairs=int((frame.status != 'ok').sum()),
                  max_absolute_error_matched=float(complete.absolute_error_matched.max()),
                  primary_fraction_error_over_01=gate.to_dict(),
                  advancement_gate_passed=bool(len(gate)==2 and (gate>=.05).all()),
                  tests_scored=False, validation_labels_used=False, training_executed=False)
    write(OUT/'audit_summary.json', status)
    print(json.dumps(status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check','run','summary'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='check':
        synthetic()
    elif args.stage=='run':
        if (OUT/'freeze.json').exists() or (OUT/'states').exists():
            raise FileExistsError('E4 already started; do not overwrite')
        checks=json.loads((OUT/'synthetic_checks.json').read_text())
        assert checks['all_passed'] and checks['code_sha256']==sha(__file__)
        assert checks['protocol_sha256']==sha(PROTOCOL)
        sources=[Path(__file__),PROTOCOL,ROOT/'src/threads.py',E1/'MANIFEST.sha256',E3/'freeze.json',OUT/'synthetic_checks.json']
        for dataset in ['ml100k','ml1m']:
            sources += [data_directory(dataset)/name for name in ['train.csv','meta.json']]
        write(OUT/'freeze.json',dict(created_utc=stamp(),sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
                                     expected_states=50,expected_pairs=3200,test_scored=False))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(real_job,r) for r in inputs()]
            for n,f in enumerate(as_completed(futures),1):
                print(json.dumps(dict(completed=n,**f.result())),flush=True)
        summarize()
    else:
        summarize()


if __name__=='__main__':
    main()
