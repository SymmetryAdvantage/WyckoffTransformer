#!/usr/bin/env python3
"""Benchmark already-scored structures on the same valid genes in every track."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import numbers
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TRACKS = ['free', 'fixed_symmetry']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-index', type=Path, required=True)
parser.add_argument('--arms', nargs='+')
parser.add_argument('--no-wandb', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
source_index = json.loads(args.source_index.read_text())
args.arms = args.arms or list(source_index['arms'])
args.output = args.output or args.source_index.parent / 'report'
spec = importlib.util.spec_from_file_location('bias_scorer', 'scripts/run_mlip_bias_genbench.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
root = Path('generated/mlip_bias_study/deps/lemat_genbench')
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == 'fbf1ba4855934acb8fe87135315504899064080d'
bench = scorer._import_runner(root)
config = bench.load_benchmark_config('comprehensive_multi_mlip_hull')
config['cache_dir'] = str(root / 'data')
config['preprocessor_config']['relax_structures'] = False
for key in ['js_distributions_file', 'mmd_values_file']:
    config[key] = str(root / config[key])

inputs, sources, novelty_cache = {}, {}, {}
for arm in args.arms:
    for track in TRACKS:
        directory = Path(source_index['arms'][arm]['outputs'][track])
        metadata_path = directory / 'metrics.json'
        metadata = json.loads(metadata_path.read_text())
        assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == source_index['arms'][arm]['metrics_sha256'][track], 'Pinned source metrics changed'
        assert metadata['arm_complete'], (arm, track)
        sources[f'{arm}/{track}'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        structures = scorer._load_checkpoint(directory / 'single_point_checkpoint.json.gz')
        assert len(structures) == len(metadata['scored_gene_ids'])
        # A documented unsupported single point remains in the full-cohort
        # export, but cannot enter a comparison requiring all three scorers.
        eligible = {}
        for gene, structure in zip(metadata['scored_gene_ids'], structures):
            record = scorer._single_point_record(gene, structure)
            if all(value is not None for values in record['by_scorer'].values() for value in values.values()) and all(
                    structure.properties.get(f'graph_embedding_{model}') is not None
                    for model in scorer.SCORING_MLIPS):
                eligible[gene] = structure
        inputs[arm, track] = eligible
        novelty_cache.update(scorer._load_checkpoint(directory / 'novelty_cache_checkpoint.json.gz'))
genes = sorted(set.intersection(*(set(rows) for rows in inputs.values())))
assert genes, 'No common valid genes'
first = next(iter(inputs.values()))
for gene in genes:
    assert all(first[gene].composition.fractional_composition == rows[gene].composition.fractional_composition
               for rows in inputs.values()), f'Composition differs across selected structures for gene {gene}'
scorer._install_novelty_acceleration(root, novelty_cache, lambda cache: None)
args.output.mkdir(parents=True, exist_ok=True)
path = args.output / 'common_cohort_metrics.json'
identity = {'source_metrics_sha256': sources, 'common_valid_gene_ids': genes,
            'benchmark_config': config, 'novelty_reference_revision': scorer.NOVELTY_REFERENCE_REVISION,
            'scoring_source_index_sha256': hashlib.sha256(args.source_index.read_bytes()).hexdigest()}
common_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
payload = {'identity': identity, 'results': {}, 'all_metrics_complete': False,
           'common_gene_compositions_match': True,
           'notes': ['All tracks use the same genes passing validity and fully scored by all three models in every source cohort.',
                     'Single points and preprocessing properties are reused; no calculator or relaxation runs.',
                     'Validity is not recomputed on this already-valid intersection.']}
if not path.exists() and not args.no_wandb:
    import wandb

    try:
        wandb.Api().artifact('symmetry-advantage/WyckoffTransformer/mlip-bias-common-checkpoint-' +
                             common_digest[:16] + ':latest', type='mlip_bias_common_checkpoint').download(root=str(args.output))
    except wandb.errors.CommError as exc:
        if 'not found' not in str(exc).lower() and 'does not exist' not in str(exc).lower():
            raise
if path.exists():
    saved = json.loads(path.read_text())
    assert saved['identity'] == identity, 'Common-cohort checkpoint inputs changed'
    payload = saved
common_run = None
if not args.no_wandb:
    import wandb

    from wyckoff_transformer.paths import wandb_dir
    common_run = wandb.init(entity='symmetry-advantage', project='WyckoffTransformer',
                           id='c' + common_digest[:7], resume='allow', dir=wandb_dir(),
                           job_type='mlip_bias_common_genbench', config={'identity': identity})
for (arm, track), records in inputs.items():
    key = f'{arm}/{track}'
    if key in payload['results'] and payload['results'][key]['metrics_complete']:
        continue
    structures = [records[gene] for gene in genes]
    results = scorer._drop_relaxation_rmsd(bench.run_remaining_benchmarks(
        structures, list(scorer.FAMILIES), config))
    complete = scorer._metrics_complete(results)
    payload['results'][key] = {'n_common_valid_genes': len(genes), 'metrics_complete': complete,
                               'results': results}
    payload['date_utc'] = datetime.now(timezone.utc).isoformat()
    payload['all_metrics_complete'] = (len(payload['results']) == len(inputs)
                                     and all(row['metrics_complete'] for row in payload['results'].values()))
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)
    if common_run is not None:
        checkpoint = wandb.Artifact('mlip-bias-common-checkpoint-' + common_digest[:16], type='mlip_bias_common_checkpoint')
        checkpoint.add_file(str(path), name=path.name)
        checkpoint.add_file(str(args.source_index), name='scoring_source_index.json')
        common_run.log_artifact(checkpoint).wait()
    print('COMMON_COHORT', key, 'genes', len(genes), 'complete', complete, flush=True)
    assert complete, f'Incomplete common-cohort metrics for {key}'
rows = []
for key, data in payload['results'].items():
    arm, track = key.split('/')
    row = {'arm': arm, 'track': track, 'common_valid_genes': len(genes), 'final_results': data['metrics_complete']}
    for family, result in data['results'].items():
        for metric, value in result.get('final_scores', {}).items():
            if isinstance(value, numbers.Real) and math.isfinite(value):
                row[f'{family}/{metric}'] = value
    rows.append(row)
columns = list(dict.fromkeys(key for row in rows for key in row))
with (args.output / 'common_cohort_metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
print('COMMON_COHORT_COMPLETE', len(rows), 'tracks', len(genes), 'genes', flush=True)

if common_run is not None:
    common_run.summary['all_metrics_complete'] = payload['all_metrics_complete']
    common_run.summary['common_valid_fully_scored_genes'] = len(genes)
    common_run.finish()
