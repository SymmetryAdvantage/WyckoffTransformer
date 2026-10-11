#!/usr/bin/env python3
"""Report full-cohort benchmarks and matched-gene energy comparisons separately."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import numbers
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.core import Composition

from wyckoff_transformer.energy_fields import EnergyField, check_compatible
from wyckoff_transformer.paths import wandb_dir

TRACKS = ['free', 'fixed_symmetry']
SCORERS = ['orb', 'mace', 'uma']
QUANTITIES = ['energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull']


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return isinstance(value, numbers.Real) and math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    index = json.loads(args.source_index.read_text())
    arms = list(index['arms'])
    args.output = args.output or args.source_index.parent / 'report'
    benchmarks, scores, compositions, source_hashes, full_rows = {}, {}, {}, {}, []
    for arm in arms:
        for track in TRACKS:
            directory = Path(index['arms'][arm]['outputs'][track])
            path = directory / 'metrics.json'
            data = json.loads(path.read_text())
            energy_path = directory / 'single_point_energies.json'
            energy_rows = json.loads(energy_path.read_text())
            benchmarks[arm, track] = data
            scores[arm, track] = {row['gene']: row['by_scorer'] for row in energy_rows}
            compositions[arm, track] = {row['gene']: Composition(row['formula']).fractional_composition
                                        for row in energy_rows}
            for source in [path, energy_path]:
                source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            row = {'arm': arm, 'track': track, 'selected_genes': data['selected_genes'],
                   'valid_genes': len(data['scored_gene_ids']), 'final_results': data['final_results'],
                   'scoring_complete': data['scoring_complete'], 'metrics_complete': data['metrics_complete'],
                   **{f'scored_{model}': count for model, count in data['single_point_coverage'].items()}}
            fd = data['results']['distribution']['evaluator_results']['FrechetDistance']['metric_results']['FrechetDistance']
            row['distribution_fd_models_computed'] = fd['uncertainties']['n_models_computed']
            for family, result in data['results'].items():
                for metric, value in result.get('final_scores', {}).items():
                    if finite(value):
                        row[f'{family}/{metric}'] = value
            full_rows.append(row)

    def compare_fields(left, right, model, quantity):
        check_compatible(EnergyField.from_dict(left['energy_fields'][model][quantity]),
                         EnergyField.from_dict(right['energy_fields'][model][quantity]),
                         'matched-gene GenBench energy comparison')

    paired_rows, paired_genes, residual_rows, residual_details = [], {}, [], {}
    for arm in arms:
        for model in SCORERS:
            per_quantity = {}
            for quantity in QUANTITIES:
                compare_fields(benchmarks[arm, 'free'], benchmarks[arm, 'fixed_symmetry'], model, quantity)
                genes = sorted(set(scores[arm, 'free']) & set(scores[arm, 'fixed_symmetry']))
                assert all(compositions[arm, 'free'][gene] == compositions[arm, 'fixed_symmetry'][gene]
                           for gene in genes), f'Paired compositions differ for {arm}'
                genes = [gene for gene in genes if all(finite(scores[arm, track][gene][model][quantity])
                                                     for track in TRACKS)]
                delta = np.array([scores[arm, 'free'][gene][model][quantity]
                                  - scores[arm, 'fixed_symmetry'][gene][model][quantity] for gene in genes])
                per_quantity[quantity] = dict(zip(genes, delta))
                paired_genes[f'{arm}/{model}/{quantity}'] = genes
                paired_rows.append({'arm': arm, 'scorer': model, 'quantity': quantity, 'n_pairs': len(genes),
                                    'mean_free_minus_fixed_ev_per_atom': float(delta.mean()) if len(delta) else None,
                                    'median_free_minus_fixed_ev_per_atom': float(np.median(delta)) if len(delta) else None,
                                    'std_free_minus_fixed_ev_per_atom': float(delta.std()) if len(delta) else None})
            # The frozen UMA methods reevaluate energy for formation/hull
            # quantities, so GPU reduction variation need not cancel exactly.
            # Preserve the outputs and quantify the residual rather than
            # replacing those evaluations with another energy sample.
            energy, formation = (per_quantity[q] for q in QUANTITIES[:2])
            residual = {gene: energy[gene] - formation[gene] for gene in energy.keys() & formation.keys()}
            values = np.array(list(residual.values()))
            residual_details[f'{arm}/{model}'] = residual
            residual_rows.append({'arm': arm, 'scorer': model, 'n_pairs': len(values),
                                  'mean_signed_residual_ev_per_atom': float(values.mean()),
                                  'median_absolute_residual_ev_per_atom': float(np.median(abs(values))),
                                  'max_absolute_residual_ev_per_atom': float(max(abs(values))),
                                  'pairs_above_1e_minus5_ev_per_atom': int((abs(values) > 1e-5).sum())})
            if model != 'uma' and max(abs(values)) > 1e-5:
                raise ValueError(f'Unexpected formation/total-energy residual for {arm}/{model}')

    common_path = args.output / 'common_cohort_metrics.json'
    common = json.loads(common_path.read_text())
    assert common['all_metrics_complete']
    benchmark_common_genes = set(common['identity']['common_valid_gene_ids'])
    common_rows, common_genes = [], {}
    for track in TRACKS:
        for model in SCORERS:
            for quantity in QUANTITIES:
                gene_sets = [{gene for gene, row in scores[arm, track].items() if finite(row[model][quantity])}
                             for arm in arms]
                genes = sorted(set.intersection(*gene_sets) & benchmark_common_genes)
                common_genes[f'{track}/{model}/{quantity}'] = genes
                reference = benchmarks[arms[0], track]
                for arm in arms:
                    compare_fields(reference, benchmarks[arm, track], model, quantity)
                    values = np.array([scores[arm, track][gene][model][quantity] for gene in genes])
                    row = {'arm': arm, 'track': track, 'scorer': model, 'quantity': quantity,
                           'n_common_genes': len(genes), 'mean_ev_per_atom': float(values.mean()) if len(values) else None,
                           'median_ev_per_atom': float(np.median(values)) if len(values) else None}
                    if quantity == 'energy_above_hull':
                        row.update(stable_fraction=float((values <= 0).mean()) if len(values) else None,
                                   below_0p1_fraction=float((values <= .1).mean()) if len(values) else None)
                    common_rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'full_cohort_metrics.csv', full_rows)
    write_csv(args.output / 'paired_free_fixed_energy_statistics.csv', paired_rows)
    write_csv(args.output / 'common_gene_energy_statistics.csv', common_rows)
    write_csv(args.output / 'formation_reference_cancellation_residuals.csv', residual_rows)
    (args.output / 'formation_reference_cancellation_residual_details.json').write_text(
        json.dumps(residual_details, indent=2) + '\n')
    assert set(common['results']) == {f'{arm}/{track}' for arm, track in benchmarks}
    assert common['identity']['source_metrics_sha256'] == {
        f'{arm}/{track}': source_hashes[str(Path(index['arms'][arm]['outputs'][track]) / 'metrics.json')]
        for arm, track in benchmarks}
    source_hashes[str(common_path)] = hashlib.sha256(common_path.read_bytes()).hexdigest()
    provenance = {'date_utc': datetime.now(timezone.utc).isoformat(),
                  'wyformer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'sources_sha256': source_hashes, 'arms': arms, 'tracks': TRACKS,
                  'scoring_run_ids': {f'{arm}/{track}': data.get('scoring_run_id') or hashlib.sha256(json.dumps(
                      {'arm': data['arm_manifest'], 'track': track}, sort_keys=True).encode()).hexdigest()[:8]
                      for (arm, track), data in benchmarks.items()},
                  'relaxation_artifacts': {arm: benchmarks[arm, TRACKS[0]]['arm_artifact'] for arm in arms},
                  'paired_gene_ids': paired_genes, 'common_gene_ids': common_genes,
                  'benchmark_common_valid_gene_ids': common['identity']['common_valid_gene_ids'],
                  'common_cohort_metrics_complete': common['all_metrics_complete'],
                  'scoring_failure_gene_ids': {f'{arm}/{track}': data['scoring_failure_gene_ids']
                                               for (arm, track), data in benchmarks.items()},
                  'energy_fields': benchmarks[arms[0], TRACKS[0]]['energy_fields'],
                  'warnings': ['Full-cohort benchmark rows have different successful-gene cohorts.',
                               'Frozen charge-validity behavior is retained, including its documented limitations.',
                               'Unsupported scorer predictions, if present, remain explicit source coverage failures; matched comparisons exclude the same genes from every track.',
                               'Relaxation snapshots are pinned after their declared recovery completion; reused scoring sources retain their verified artifact and output hashes.',
                               'Common-gene energy tables use the benchmark intersection, with separate compatible scorers and quantities.',
                               'Common-cohort benchmarks use the intersection of valid, fully scored genes across every arm and both tracks.',
                               'Frozen LeMat source records an unresolved UMA energy/hull reference discrepancy.',
                               'Large UMA noble-gas predictions persist in a fresh process and remain unfiltered.',
                               'GPU reductions can vary between scoring workers; benchmark thresholds are unchanged.',
                               'Frozen UMA formation/hull helpers reevaluate energy; cancellation residuals are measured in the accompanying tables and predictions remain unchanged.',
                               'Upstream ensemble descriptors are protocol outputs; physical energy comparisons use separate scorers.'],
                  'all_tracks_final': all(data['final_results'] for data in benchmarks.values())}
    (args.output / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    shutil.copyfile(args.source_index, args.output / 'scoring_source_index.json')
    provenance['scoring_source_index_sha256'] = hashlib.sha256(args.source_index.read_bytes()).hexdigest()
    provenance['snapshot_namespace'] = index['snapshot_namespace']
    source_files = ['scripts/run_mlip_bias_genbench.py', 'scripts/compute_mlip_bias_common_genbench.py',
                    'scripts/report_mlip_bias_genbench.py', 'scripts/verify_mlip_bias_genbench_report.py',
                    'scripts/platforms/zeus/watch_mlip_bias_genbench.py',
                    'scripts/platforms/zeus/report_updated_mlip_bias_genbench.sh']
    provenance['source_file_sha256'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                         for path in source_files}
    (args.output / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    setup = Path('generated/mlip_bias_study/setup')
    shutil.copyfile(setup / 'genbench_environment_20261006.json', args.output / 'historical_genbench_environment_20261006.json')
    packages = {name: importlib.metadata.version(name) for name in
                ['torch', 'numpy', 'scipy', 'pymatgen', 'wandb', 'fairchem-core', 'orb-models', 'mace-torch']}
    (args.output / 'report_execution_environment.json').write_text(json.dumps(
        {'date_utc': provenance['date_utc'], 'packages': packages,
         'wyformer_commit': provenance['wyformer_commit'],
         'source_file_sha256': provenance['source_file_sha256'],
         'note': 'Historical full setup/checkpoint provenance is included separately; this records current reporting packages and source.'},
        indent=2) + '\n')
    for filename in ['genbench_hull_provenance_20261006.json',
                     'genbench_novelty_reference_20261006.json', 'genbench_reference_alignment_20261006.json']:
        shutil.copyfile(setup / filename, args.output / filename)
    subprocess.run(['.venv/bin/python', 'scripts/verify_mlip_bias_genbench_report.py',
                    '--source-index', str(args.source_index), '--output', str(args.output)],
                   check=True, stdout=subprocess.DEVNULL)
    if not args.no_wandb:
        import wandb
        digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
        run = wandb.init(entity='symmetry-advantage', project='WyckoffTransformer', id=digest[:8], resume='allow',
                         job_type='mlip_bias_analysis', dir=str(wandb_dir()), config={'arms': arms, 'tracks': TRACKS})
        artifact = wandb.Artifact('mlip-bias-genbench-comparison-' + digest[:16], type='mlip_bias_metrics_report',
                                  metadata={'date_utc': provenance['date_utc'], 'all_tracks_final': provenance['all_tracks_final'],
                                            'common_cohort_metrics_complete': common['all_metrics_complete']})
        for file in args.output.iterdir():
            if file.is_file(): artifact.add_file(str(file), name=file.name)
        artifact.add_file(__file__, name='report_mlip_bias_genbench.py')
        artifact.add_file('scripts/compute_mlip_bias_common_genbench.py', name='compute_mlip_bias_common_genbench.py')
        artifact.add_file('docs/lemat_genbench_charge_validity_bugs.md', name='charge_validity_limitations.md')
        artifact.add_file('scripts/verify_mlip_bias_genbench_report.py', name='verify_mlip_bias_genbench_report.py')
        run.log_artifact(artifact).wait()
        import pandas as pd
        run.log({name: wandb.Table(dataframe=pd.read_csv(args.output / (name + '.csv')))
                 for name in ['full_cohort_metrics', 'paired_free_fixed_energy_statistics',
                              'common_gene_energy_statistics', 'common_cohort_metrics',
                              'formation_reference_cancellation_residuals']})
        run.summary['all_tracks_final'] = provenance['all_tracks_final']
        run.summary['common_cohort_metrics_complete'] = common['all_metrics_complete']
        run.summary['common_valid_fully_scored_genes'] = len(benchmark_common_genes)
        run.summary['documented_scoring_failures'] = sum(len(genes) for failures in
            provenance['scoring_failure_gene_ids'].values() for genes in failures.values())
        print('REPORT_RUN', run.id, 'ARTIFACT', artifact.qualified_name)
        run.finish()
    print(json.dumps({'paired_rows': len(paired_rows), 'common_rows': len(common_rows),
                      'all_tracks_final': provenance['all_tracks_final']}, indent=2))


if __name__ == '__main__':
    main()
