"""Add row formation energies to a sibling tensor cache, preserving every token id.

The parent tensors and dataset rows must agree on space group, hull distance and
gene-minimum energy before any field is attached. The parent is never changed.
This copies the flat safetensors payload to avoid expanding millions of ragged
augmentation variants into Python objects just to append one scalar column.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from wyckoff_transformer.dataset_cache import (
    build_info,
    dataset_cache_dir,
    load_split,
    provenance,
)
from wyckoff_transformer.dataset_manifest import (
    check_cache_matches,
    load_manifest,
    refuse_if_obsolete_dataset,
)
from wyckoff_transformer.tokenization import (
    _TENSOR_CACHE_FORMAT_KEY,
    _TENSOR_CACHE_FORMAT_VERSION,
    _TENSOR_CACHE_STRUCTURE_KEY,
)
from wyckoff_transformer.wyckoff_processor import WyckoffProcessor


def prepare(dataset: str, config_path: Path) -> Path:
    refuse_if_obsolete_dataset(dataset, 'preparing formation-energy tensors')
    config = OmegaConf.load(config_path)
    root = dataset_cache_dir(dataset)
    source = root / 'tensors' / 'der_tokenizer_v1.safetensors'
    destination = root / 'tensors' / f'{config.name}.safetensors'
    processor_path = root / 'tokenisers' / f'{config.name}.json'
    record_path = destination.with_suffix('.build.json')
    if any(p.exists() for p in (destination, processor_path, record_path)):
        raise FileExistsError(f'Refusing to overwrite an existing cache: {destination}')
    processor = WyckoffProcessor.from_pretrained(root / 'tokenisers' / 'der_tokenizer_v1.json')
    expected = OmegaConf.to_container(processor.config)
    expected['name'] = config.name
    expected['sequence_fields']['no_processing'].append('formation_energy_per_atom')
    if expected != OmegaConf.to_container(config):
        raise ValueError('The sibling must differ only by name and formation_energy_per_atom')
    manifest = load_manifest(dataset)
    field = manifest.resolve('formation_energy_per_atom')
    records = build_info(root)
    for info in records.values():
        check_cache_matches(dataset, ((info or {}).get('options') or {}).get('fields'))
    with safe_open(source, framework='pt', device='cpu') as handle:
        metadata = handle.metadata()
        if metadata[_TENSOR_CACHE_FORMAT_KEY] != _TENSOR_CACHE_FORMAT_VERSION:
            raise ValueError('Unsupported tensor cache format')
        structure = json.loads(metadata[_TENSOR_CACHE_STRUCTURE_KEY])
        payload = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
        sizes = {}
        for split, node in structure['items'].items():
            fields = node['items']
            frame = load_split(root, split, columns=[
                'formation_energy_per_atom', 'spacegroup_number',
                'energy_above_hull', 'gene_min_formation_energy_per_atom',
            ])
            frame = frame[frame.spacegroup_number.isin(processor.tokenisers['spacegroup_number'])]
            existing_sg = payload[fields['spacegroup_number']['key']]
            encoded_sg = processor.tokenisers['spacegroup_number'].encode_spacegroups(
                frame.spacegroup_number, dtype=existing_sg.dtype)
            if not torch.equal(encoded_sg, existing_sg):
                raise ValueError(f'{dataset}/{split}: space group or row order differs')
            for name in ('energy_above_hull', 'gene_min_formation_energy_per_atom'):
                existing = payload[fields[name]['key']]
                expected_values = torch.tensor(frame[name].to_numpy(), dtype=existing.dtype)
                torch.testing.assert_close(existing, expected_values, rtol=0, atol=0,
                                           equal_nan=True, msg=f'{dataset}/{split}/{name}')
            energy = frame.formation_energy_per_atom.to_numpy()
            if not np.isfinite(energy).all():
                raise ValueError(f'{dataset}/{split}: missing or non-finite formation energy')
            key = f'formation_energy_{split}'
            payload[key] = torch.tensor(energy, dtype=torch.float32)
            fields['formation_energy_per_atom'] = {'type': 'tensor', 'key': key}
            sizes[split] = len(frame)
            print(f'{dataset}/{split}: aligned {len(frame):,} rows, '
                  f'formation energy [{energy.min():.3f}, {energy.max():.3f}] eV/atom', flush=True)
        metadata[_TENSOR_CACHE_STRUCTURE_KEY] = json.dumps(structure, separators=(',', ':'))
        record = provenance(
            'scripts/prepare_formation_energy_token_cache.py', dataset=dataset,
            source_tokeniser='der_tokenizer_v1', target_tokeniser=config.name,
            added_field={field.name: field.to_record()}, source_build=records, split_rows=sizes,
            alignment=['spacegroup_number', 'energy_above_hull', 'gene_min_formation_energy_per_atom'],
        )
        metadata['wyformer_build'] = json.dumps(record)
        temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
        save_file(payload, temporary, metadata=metadata)
        temporary.replace(destination)
    processor.config = config
    processor.save_pretrained(processor_path)
    record_path.write_text(json.dumps(record, indent=2))
    return record_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--config', type=Path,
                        default=Path('yamls/tokenisers/der_tokenizer_v1_formation_energy.yaml'))
    args = parser.parse_args()
    print(prepare(args.dataset, args.config), flush=True)
