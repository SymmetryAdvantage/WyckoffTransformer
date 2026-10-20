"""Run the six formula CSP models in order on both Hera GPUs, with W&B records.

Start from the checkout root with its .venv/bin/python. The manifest records the
W&B selection and independent model order; a restart resumes checkpointed jobs
and skips finished runs. An existing run without resume state is never restarted.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from safetensors import safe_open

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli.resume_probe import wandb_checkpoint_exists
from wyckoff_transformer.dataset_cache import dataset_cache_dir
from wyckoff_transformer.paths import runs_root, wandb_dir
from wyckoff_transformer.tokenization import _TENSOR_CACHE_STRUCTURE_KEY
from wyckoff_transformer.trainer import CHECKPOINT_FILENAME


def validate_inputs(manifest):
    """Check every batch against the cached split before starting the queue."""
    for job in manifest['jobs']:
        config = OmegaConf.load(job['config'])
        trainer = config.model.WyckoffTrainer_args
        path = (dataset_cache_dir(job['dataset']) / 'tensors'
                / f'{config.tokeniser.name}.safetensors')
        with safe_open(path, framework='pt', device='cpu') as handle:
            structure = json.loads(handle.metadata()[_TENSOR_CACHE_STRUCTURE_KEY])
            for split, node in structure['items'].items():
                key = node['items']['pure_sequence_length']['key']
                count = handle.get_slice(key).get_shape()[0]
                batch = trainer.get(f'{split}_batch_size')
                if batch is not None and batch > count:
                    raise ValueError(f'{job["name"]}/{split}: batch {batch} exceeds {count} rows')
                if split == 'train' and (batch is None or batch % manifest['world_size']):
                    raise ValueError(f'{job["name"]}: training batch must divide over the GPUs')
        print(f'Validated input batches for {job["name"]}', flush=True)


def find_run(api, run_id):
    try:
        return api.run(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}')
    except wandb.errors.CommError as exc:
        message = str(exc).lower()
        if 'could not find run' in message and 'not found' in message:
            return None
        raise


def checkpoint_epoch(run_id):
    path = runs_root() / run_id / CHECKPOINT_FILENAME
    if not path.is_file():
        return -1
    return int(torch.load(path, map_location='cpu', weights_only=True)['epoch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('yamls/campaigns/csp_formula_hera_20261016.yaml'))
    parser.add_argument('--evidence', type=Path, required=True,
                        help='JSON snapshot of W&B run configurations and summaries')
    args = parser.parse_args()
    manifest = OmegaConf.to_container(OmegaConf.load(args.manifest))
    validate_inputs(manifest)
    campaign_id = manifest['campaign_id']
    directory = runs_root() / campaign_id
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES'] = manifest['gpu_ids']
    os.environ['WANDB_MODE'] = 'online'
    api = wandb.Api(timeout=60)
    old_campaign = find_run(api, campaign_id)
    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                    id=campaign_id, name=campaign_id, job_type='campaign',
                    group=campaign_id, dir=wandb_dir(), config=manifest,
                    resume='must' if old_campaign else None) as campaign:
        campaign.config.update(manifest, allow_val_change=True)
        selection = wandb.Artifact(f'{campaign_id}_selection', type='campaign')
        selection.add_file(args.manifest, name='manifest.yaml')
        selection.add_file(args.evidence, name='wandb_runs.json')
        for job in manifest['jobs']:
            selection.add_file(job['config'], name=job['config'])
        for path in [
            'yamls/tokenisers/der_tokenizer_v1_formation_energy.yaml',
            'scripts/platforms/hera/train_csp_campaign.py',
            'scripts/prepare_formation_energy_token_cache.py',
        ]:
            selection.add_file(path, name=path)
        campaign.log_artifact(selection).wait()
        status = {}
        for position, job in enumerate(manifest['jobs']):
            job_id = job.get('run_id', f'{campaign_id}-{job["name"]}')
            status[job['name']] = {'id': job_id, 'state': 'queued', 'returncode': None}
        campaign.summary['jobs'] = status
        for position, job in enumerate(manifest['jobs']):
            run_id = status[job['name']]['id']
            log_path = directory / f'{job["name"]}.log'
            stalled = 0
            for attempt in range(1, 31):
                api.flush()
                previous = find_run(api, run_id)
                if previous is not None and previous.state == 'finished':
                    status[job['name']]['state'] = 'finished'
                    campaign.summary['jobs'] = status
                    break
                epoch_before = checkpoint_epoch(run_id)
                resume = epoch_before >= 0
                if not resume and previous is not None:
                    remote = wandb_checkpoint_exists(run_id, WANDB_ENTITY, WANDB_PROJECT)
                    if remote is not True:
                        raise RuntimeError(
                            f'{run_id} exists without accessible resume state; refusing to restart')
                    resume = True
                config = Path(job['config'])
                child_config = OmegaConf.load(config)
                tokeniser = child_config.tokeniser.name
                cache = dataset_cache_dir(job['dataset'])
                if not (cache / 'tensors' / f'{tokeniser}.safetensors').is_file():
                    raise FileNotFoundError(f'Missing input tensors for {job["dataset"]}/{tokeniser}')
                command = [
                    '.venv/bin/python', '-u', '-m', 'torch.distributed.run',
                    '--standalone', '--nproc-per-node', str(manifest['world_size']),
                    'scripts/train.py', str(config), job['dataset'], 'cuda',
                    '--torch-num-thread', '8',
                ]
                if resume:
                    command.extend(['--resume', run_id])
                env = os.environ.copy()
                env.update(WANDB_RUN_ID=run_id, WANDB_NAME=run_id,
                           WANDB_RUN_GROUP=campaign_id, WANDB_JOB_TYPE='train',
                           WANDB_TAGS='csp,formula,hera,'+job['name'],
                           OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                           WYFORMER_GIT_COMMIT=subprocess.check_output(
                               ['git', 'rev-parse', 'HEAD'], text=True).strip())
                print(f'{time.strftime("%F %T")} {job["name"]}, attempt {attempt}: '
                      f'{"resuming" if resume else "starting"} {run_id}', flush=True)
                status[job['name']].update(state='running', attempt=attempt)
                campaign.summary['jobs'] = status
                campaign.log({'active_job_index': position, 'attempt': attempt})
                with log_path.open('a') as logfile:
                    child = subprocess.Popen(command, env=env, stdout=logfile,
                                             stderr=subprocess.STDOUT)
                    campaign.save(str(log_path), base_path=str(directory), policy='live')
                    while child.poll() is None:
                        time.sleep(30)
                        campaign.log({'active_job_index': position, 'child_pid': child.pid})
                    returncode = child.returncode
                campaign.save(str(log_path), base_path=str(directory), policy='now')
                if returncode == 0:
                    status[job['name']]['state'] = 'finished'
                    campaign.summary['jobs'] = status
                    break
                epoch_after = checkpoint_epoch(run_id)
                stalled = stalled + 1 if epoch_after <= epoch_before else 0
                print(f'{run_id} exited {returncode}, checkpoint epoch {epoch_after}', flush=True)
                if stalled >= 3 or (epoch_after < 0 and not resume):
                    status[job['name']].update(state='failed', returncode=returncode)
                    campaign.summary['jobs'] = status
                    raise RuntimeError(f'{run_id} failed without checkpoint progress; see {log_path}')
                time.sleep(30)
            else:
                raise RuntimeError(f'Attempt budget exhausted for {run_id}')
        campaign.summary['complete'] = True
        print('All six CSP training runs finished.', flush=True)


if __name__ == '__main__':
    main()
