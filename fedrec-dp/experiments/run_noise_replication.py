"""E3 independent-data diagnostic; never scores a test target or edits E1/E2."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.threads import force_env
force_env()
import argparse
import copy
import hashlib
import json
import re
import time
import zipfile
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import src.effective_noise as mechanism
from src.effective_noise import EffectiveNoiseBPR, bounded_popularity, analytic_gaussian_sigma
from src.data import chronological_leave_two_out, validate_split, save_split, split_fingerprint, RAW_COLUMNS
from src.evaluate import evaluate, rank_targets, target_items, exclusion_lists
from src.metrics import ndcg_from_rank
from src.federated import sample_clients
from src.threads import enforce_and_report
import probe_noise_memory as e2
from probe_noise_memory_aligned import align

OUT = ROOT / 'results/extensions/noise_replication_v1'
DATA = ROOT / 'data/extensions/ml-1m_noise_v1'
CHECKPOINTS = ROOT / 'checkpoints/extensions/noise_replication_v1'
PROTOCOL = ROOT / 'docs/extensions/NOISE_REPLICATION_PROTOCOL.md'
E1 = ROOT / 'results/extensions/effective_noise_v1'
SEEDS = [42, 123, 2026, 7, 99]
UNITS = ['full', 'fixed_r8', 'two_r8']
LEVELS = ['eps1', 'eps2']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if DATA.exists():
        raise FileExistsError(DATA)
    archive = Path('/tmp/cs670_ml-1m.zip')
    published = Path('/tmp/cs670_ml-1m.zip.md5').read_text()
    expected = re.search(r'\b[0-9a-f]{32}\b', published).group()
    assert hashlib.md5(archive.read_bytes()).hexdigest() == expected
    raw = DATA / 'raw'
    raw.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            p = Path(name)
            assert not p.is_absolute() and '..' not in p.parts
        z.extractall(raw)
    ratings = pd.read_csv(raw / 'ml-1m/ratings.dat', sep='::', names=RAW_COLUMNS,
                          engine='python', dtype='int64')
    assert not ratings.duplicated(['user_id', 'item_id']).any()
    split, stats = chronological_leave_two_out(ratings, 4, 3, 2026)
    validate_split(split)
    save_split(split, DATA / 'processed')
    observer = np.sort(np.random.default_rng([6708, 1000]).choice(split['n_users'], 128, replace=False))
    meta = dict(created_utc=stamp(), source='https://files.grouplens.org/datasets/movielens/ml-1m.zip',
                archive_md5=expected, archive_sha256=sha(archive), protocol_sha256=sha(PROTOCOL),
                raw_sha256={str(p.relative_to(DATA)): sha(p) for p in raw.rglob('*') if p.is_file()},
                split_sha256={str(p.relative_to(DATA)): sha(p) for p in (DATA / 'processed').glob('*')},
                split_fingerprint=split_fingerprint(split), n_users=split['n_users'], n_items=split['n_items'],
                train_positives=len(split['train']), validation_targets=len(split['validation']),
                test_targets=len(split['test']), statistics=stats, observer_users=observer.tolist(), test_scored=False)
    write(DATA / 'dataset.json', meta)
    write(OUT / 'dataset.json', meta)
    print(json.dumps({k: meta[k] for k in ('n_users', 'n_items', 'train_positives', 'split_fingerprint')}))


def load_data(dataset):
    directory = DATA / 'processed' if dataset == 'ml1m' else ROOT / 'data/processed/ml-100k'
    split = json.loads((directory / 'meta.json').read_text())
    # The real test file is deliberately never read during training/probes.
    for name in ('train', 'validation', 'history'):
        split[name] = pd.read_csv(directory / f'{name}.csv', dtype='int64')
    code = 1000 if dataset == 'ml1m' else 100
    users = np.sort(np.random.default_rng([6708, code]).choice(split['n_users'], 128, replace=False))
    return split, users


def old_records():
    return [r for r in json.loads((E1 / 'final_records.json').read_text())
            if r['unit'] in UNITS and r['level'] in LEVELS and r['seed'] in SEEDS]


def score(model, split):
    p, q = model.P.astype(np.float64), model.Q.astype(np.float64)
    return evaluate(lambda users: p[users] @ q.T, split, 'validation')


def train_job(record):
    threads = enforce_and_report()
    start = time.perf_counter()
    split, _ = load_data('ml1m')
    c = copy.deepcopy(record['settings'])
    c['score_rounds'] = []
    tag = f"ml1m_{record['unit']}_{record['level']}_seed{record['seed']}"
    path = OUT / 'training' / f'{tag}.json'
    if path.exists():
        raise FileExistsError(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = EffectiveNoiseBPR(split['train'], split['n_users'], split['n_items'], c)
    result = dict(dataset='ml1m', unit=record['unit'], level=record['level'], seed=record['seed'],
                  settings=c, tag=tag, status='ok', created_utc=stamp(), threads=threads)
    try:
        for _ in range(50):
            model.run_round()
        e2.finite(model)
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        checkpoint = CHECKPOINTS / f'{tag}.npz'
        np.savez(checkpoint, **model.state())
        result.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                      round=model.round, total_bytes=model.total_bytes,
                      final_user_norm_max=float(np.linalg.norm(model.P.astype(np.float64), axis=1).max()))
        result['validation'], ranks = score(model, split)
        pd.DataFrame({'user': np.arange(len(ranks)), 'rank': ranks}).to_csv(
            OUT / 'training' / f'{tag}_validation.csv', index=False)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        result.update(status='failed', failure=repr(exc), round=model.round)
    result['wall_time_s'] = time.perf_counter() - start
    write(path, result)
    return result


@contextmanager
def balancing(mode):
    original = mechanism.balance
    if mode == 'none':
        mechanism.balance = lambda a, b: (a, b)
    try:
        yield
    finally:
        mechanism.balance = original


def observe(branch, reference, context, phase, lag, name, exposed, pulse_rms):
    e2.finite(branch)
    e2.finite(reference)
    users, targets, excluded = context
    p, q = reference.P[users].astype(np.float64), reference.Q.astype(np.float64)
    dp = branch.P[users].astype(np.float64) - p
    dq = branch.Q.astype(np.float64) - q
    s0, s1 = p @ q.T, (p + dp) @ (q + dq).T
    ds = s1 - s0
    direct, local, interaction = p @ dq.T, dp @ q.T, dp @ dq.T
    closure = float(np.max(np.abs(ds - direct - local - interaction)))
    assert closure < 1e-10 * max(1., float(np.max(np.abs(s1))))
    ranks0 = rank_targets(s0, targets, excluded)
    ranks1 = rank_targets(s1, targets, excluded)
    rms = float(np.sqrt(np.mean(ds ** 2)))
    values0, values1 = ndcg_from_rank(ranks0), ndcg_from_rank(ranks1)
    # Stability candidates deliberately match E2; validation uses the filter above.
    s0[reference.pos_mask[users]], s1[reference.pos_mask[users]] = -np.inf, -np.inf
    top0 = np.argsort(-s0, axis=1, kind='stable')[:, :10]
    top1 = np.argsort(-s1, axis=1, kind='stable')[:, :10]
    churn = 1 - (top0[:, :, None] == top1[:, None, :]).any(axis=2).sum(axis=1) / 10
    row = dict(phase=phase, lag=lag, branch=name, score_rms=rms,
               score_ratio_to_pulse=rms / pulse_rms if pulse_rms else 0.,
               q_difference_fro=float(np.linalg.norm(dq)), p_difference_fro=float(np.linalg.norm(dp)),
               top10_churn=float(churn.mean()), validation_ndcg_reference=float(values0.mean()),
               validation_ndcg_branch=float(values1.mean()), validation_ndcg_difference=float((values1-values0).mean()),
               closure_max_abs=closure, max_user_norm=float(np.linalg.norm(branch.P.astype(np.float64), axis=1).max()))
    for label, mask in [('exposed', exposed[users]), ('unexposed', ~exposed[users])]:
        row[f'users_{label}'] = int(mask.sum())
        if mask.any():
            row[f'top10_churn_{label}'] = float(churn[mask].mean())
            row[f'score_rms_{label}'] = float(np.sqrt(np.mean(ds[mask] ** 2)))
    return row


def probe(model, context, dataset_code, replicate, coupling, balance_mode, alpha=1., checks=False, rows=None):
    rows = [] if rows is None else rows
    reference = copy.deepcopy(model)
    seed = model.settings['seed']
    for key, code in [('sampling_rng', 6711), ('client_rng', 6712), ('noise_rng', 6713)]:
        setattr(reference, key, np.random.default_rng([code, dataset_code, seed, replicate]))
    reference.settings['score_rounds'] = []
    reference.round = 0
    retained = copy.deepcopy(reference)
    c = model.settings
    tau = c['server_lr'] * c['sigma'] * c['clip_norm'] / (c['q'] * model.n_users)
    noise = np.random.default_rng([6710, dataset_code, seed, replicate]).standard_normal(model.D, dtype=np.float32)
    noise *= np.float32(alpha * tau)
    if alpha:
        if retained.method == 'full':
            retained._Q += noise.reshape(retained._Q.shape)
        else:
            n = retained.A.size
            retained.A += noise[:n].reshape(retained.A.shape)
            if retained.method == 'two':
                retained.B += noise[n:].reshape(retained.B.shape)
                if balance_mode == 'svd':
                    retained.A, retained.B = mechanism.balance(retained.A, retained.B)
    selected = sample_clients(copy.deepcopy(reference.sampling_rng), model.n_users, c['q'])
    exposed = np.zeros(model.n_users, dtype=bool)
    exposed[selected] = True
    row = observe(retained, reference, context, 'pulse', -1, 'retained', exposed, None)
    pulse_rms = row['score_rms']
    row['score_ratio_to_pulse'] = 1. if pulse_rms else 0.
    rows.append(row)
    errors = []
    if coupling == 'aligned':
        errors.append(align(retained, reference)[0])
    with balancing(balance_mode):
        reference.run_round()
        retained.run_round()
    e2.assert_rng_equal(reference, retained)
    rows.append(observe(retained, reference, context, 'before_reset', 0, 'retained', exposed, pulse_rms))
    shared, local, both = [copy.deepcopy(retained) for _ in range(3)]
    e2.shared_copy(shared, reference)
    local.P = reference.P.copy()
    e2.shared_copy(both, reference)
    both.P = reference.P.copy()
    e2.assert_equal(both, reference)
    branches = dict(retained=retained, shared_reset=shared, local_reset=local)
    for lag in range(11):
        if lag:
            if coupling == 'aligned':
                for branch in branches.values():
                    errors.append(align(branch, reference)[0])
            with balancing(balance_mode):
                reference.run_round()
                for branch in branches.values():
                    branch.run_round()
                    e2.assert_rng_equal(branch, reference)
                if checks:
                    both.run_round()
                    e2.assert_equal(both, reference)
        if alpha == 0:
            for branch in branches.values():
                e2.assert_equal(branch, reference)
        if lag in (0, 1, 4, 10):
            for name, branch in branches.items():
                row = observe(branch, reference, context, 'after_reset', lag, name, exposed, pulse_rms)
                if lag == 0 and name == 'shared_reset':
                    assert row['q_difference_fro'] == 0
                if lag == 0 and name == 'local_reset':
                    assert row['p_difference_fro'] == 0
                rows.append(row)
    for row in rows:
        row['alignment_product_error_max'] = max(errors, default=0.)
    return rows


def check():
    enforce_and_report()
    train = pd.DataFrame({'user': np.repeat(np.arange(8), 3),
                          'item': [(u * 2 + j) % 20 for u in range(8) for j in range(3)]})
    c = dict(method='two', rank=2, dim=4, init_std=.01, seed=42, public_basis_seed=314159,
             q=.5, local_lr=.1, local_epochs=2, reg=1e-5, server_lr=.1, clip_norm=1., sigma=.2,
             score_rounds=[], pair_count=32)
    context = np.arange(8), np.array([(u * 2 + 4) % 20 for u in range(8)]), [np.array([], dtype=int)] * 8
    records = []
    for method, balance_modes, couplings in [('full', ['svd'], ['raw']), ('fixed', ['svd'], ['raw']),
                                             ('two', ['svd', 'none'], ['raw', 'aligned'])]:
        model = EffectiveNoiseBPR(train, 8, 20, {**c, 'method': method})
        for b in balance_modes:
            for coupling in couplings:
                for amplitude in [0., 1.]:
                    rows = probe(model, context, 0, 0, coupling, b, amplitude, checks=True)
                    records.append(dict(method=method, balancing=b, coupling=coupling, alpha=amplitude,
                                        passed=True, observations=len(rows)))
    # The context wrapper leaves the original balanced round exactly unchanged.
    original = EffectiveNoiseBPR(train, 8, 20, c)
    wrapped = copy.deepcopy(original)
    original.run_round()
    with balancing('svd'):
        wrapped.run_round()
    e2.assert_equal(original, wrapped)
    write(OUT / 'synthetic_checks.json', dict(created_utc=stamp(), original_round_identity=True, records=records))
    print(json.dumps({'synthetic_cases': len(records), 'all_passed': True}))


def freeze():
    path = OUT / 'freeze.json'
    if path.exists():
        raise FileExistsError('E3 already started; existing outputs cannot be overwritten')
    checks = json.loads((OUT / 'synthetic_checks.json').read_text())
    assert all(r['passed'] for r in checks['records'])
    accounting = pd.read_csv(E1 / 'accounting.csv')
    records = old_records()
    assert len(records) == 30
    for record in records:
        a = accounting[(accounting.mechanism == 'training') & (accounting.target_epsilon == int(record['level'][3:]))].iloc[0]
        assert record['settings']['sigma'] == a.sigma and a.q == .1 and a['T'] == 50
    sources = [Path(__file__), PROTOCOL, ROOT / 'experiments/probe_noise_memory.py',
               ROOT / 'experiments/probe_noise_memory_aligned.py']
    sources += [ROOT / f'src/{name}.py' for name in ('effective_noise','lowrank','federated','data','evaluate','privacy','metrics','threads')]
    write(path, dict(created_utc=stamp(), sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                     dataset_sha256=sha(DATA / 'dataset.json'), input_e1_manifest_sha256=sha(E1 / 'MANIFEST.sha256'),
                     training_runs=30, probes=200, test_scored=False))


def popularity():
    split, _ = load_data('ml1m')
    counts = bounded_popularity(split['train'], split['n_users'], split['n_items'])
    records = []
    for level in LEVELS:
        epsilon = int(level[3:])
        sigma = analytic_gaussian_sigma(epsilon, 1e-5)
        for seed in SEEDS:
            noise = np.random.default_rng([6716, seed, epsilon]).normal(size=split['n_items'])
            value = counts + np.sqrt(20) * sigma * noise
            summary, _ = evaluate(lambda users: np.broadcast_to(value, (len(users), len(value))), split, 'validation')
            records.append(dict(unit='dp_popularity', level=level, seed=seed, sigma=sigma, validation_ndcg=summary['ndcg@10']))
    pd.DataFrame(records).to_csv(OUT / 'popularity_validation.csv', index=False)


def probe_job(job):
    record, dataset, coupling, balance_mode, replicate = job
    enforce_and_report()
    tag = f"{dataset}_{record['unit']}_{record['level']}_seed{record['seed']}_{balance_mode}_{coupling}_rep{replicate}"
    path = OUT / 'probes' / f'{tag}.json'
    if path.exists():
        raise FileExistsError(tag)
    result = dict(tag=tag, dataset=dataset, unit=record['unit'], level=record['level'], seed=record['seed'],
                  coupling=coupling, balancing=balance_mode, replicate=replicate, created_utc=stamp(),
                  status='ok', observations=[])
    if record.get('status', 'ok') != 'ok':
        result.update(status='unavailable_starting_state', failure=record.get('failure'))
        write(path, result)
        return {k: result[k] for k in ('tag', 'status')}
    start = time.perf_counter()
    split, users = load_data(dataset)
    all_exclusions = exclusion_lists(split, 'validation')
    context = users, target_items(split, 'validation')[users], [all_exclusions[u] for u in users]
    settings = copy.deepcopy(record['settings'])
    settings['score_rounds'] = []
    model = EffectiveNoiseBPR(split['train'], split['n_users'], split['n_items'], settings)
    checkpoint = ROOT / record['checkpoint']
    expected = record['checkpoint_sha256'] if dataset == 'ml1m' else record['artifact_hashes'][record['checkpoint']]
    assert sha(checkpoint) == expected
    with np.load(checkpoint, allow_pickle=False) as values:
        model.load_state({k: values[k] for k in model.state()})
    result.update(checkpoint=record['checkpoint'], checkpoint_sha256=expected, observer_users=users.tolist())
    try:
        probe(model, context, 1000 if dataset == 'ml1m' else 100, replicate, coupling, balance_mode,
              rows=result['observations'])
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
        result.update(status='failed', failure=repr(exc))
    result['wall_time_s'] = time.perf_counter() - start
    write(path, result)
    return {k: result[k] for k in ('tag', 'status', 'wall_time_s')}


def summarize():
    inventory, observations = [], []
    keys = ['dataset', 'unit', 'level', 'balancing', 'coupling', 'phase', 'lag', 'branch']
    for path in sorted((OUT / 'probes').glob('*.json')):
        r = json.loads(path.read_text())
        rows = r.pop('observations')
        inventory.append({k: v for k, v in r.items() if k != 'observer_users'})
        for row in rows:
            observations.append({**{k: r[k] for k in keys if k in r}, 'seed': r['seed'],
                                 'replicate': r['replicate'], 'status': r['status'], **row})
    pd.DataFrame(inventory).to_csv(OUT / 'inventory.csv', index=False)
    data = pd.DataFrame(observations)
    data.to_csv(OUT / 'observations.csv', index=False, float_format='%.12g')
    complete = data[data.status == 'ok']
    columns = [c for c in complete.select_dtypes(include='number') if c not in keys + ['seed', 'replicate']]
    seeds = complete.groupby(keys + ['seed'])[columns].mean().reset_index()
    seeds.to_csv(OUT / 'by_seed.csv', index=False, float_format='%.12g')
    summary = seeds.groupby(keys)[columns].agg(['mean', 'median', 'std', 'min', 'max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT / 'summary.csv', float_format='%.12g')
    print(json.dumps(dict(probes=len(inventory), observations=len(observations),
                         failures=sum(r['status'] != 'ok' for r in inventory))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'check', 'train', 'probe', 'summary'])
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'check':
        check()
    elif args.stage == 'train':
        freeze()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(train_job, r) for r in old_records()]
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                print(json.dumps(dict(completed=len(results), tag=r['tag'], status=r['status'], wall_time_s=r['wall_time_s'])), flush=True)
        write(OUT / 'training_records.json', results)
        popularity()
    elif args.stage == 'probe':
        declared = json.loads((OUT / 'freeze.json').read_text())
        for path, expected in declared['sources'].items():
            assert sha(ROOT / path) == expected
        if (OUT / 'probes').exists():
            raise FileExistsError('E3 probes already started; do not overwrite')
        records = [(r, 'ml100k') for r in old_records() if r['unit'] == 'two_r8']
        records += [(r, 'ml1m') for r in json.loads((OUT / 'training_records.json').read_text())]
        jobs = []
        for r, dataset in records:
            modes = [('svd','raw'), ('svd','aligned'), ('none','raw'), ('none','aligned')] if r['unit'] == 'two_r8' else [('svd','raw')]
            jobs += [(r,dataset,c,b,rep) for b,c in modes for rep in range(2)]
        assert len(jobs) == 200
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(probe_job, job) for job in jobs]
            for n, f in enumerate(as_completed(futures), 1):
                print(json.dumps(dict(completed=n, **f.result())), flush=True)
        summarize()
    else:
        summarize()


if __name__ == '__main__':
    main()
