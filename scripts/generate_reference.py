import json
import torch
from pathlib import Path
from omegaconf import OmegaConf
from wyckoff_transformer.trainer import WyckoffTrainer
from wyckoff_transformer.evaluation.cdvae_metrics import Crystal, GenEval

def main():
    run_path = Path("/home/kna/WyckoffTransformer/runs/ioi8tycx")
    config = OmegaConf.load(run_path / "config.yaml")

    trainer = WyckoffTrainer.from_config(
        config,
        device=torch.device("cpu"),
        use_cached_tensors=False,
        run_path=run_path,
        load_datasets=True
    )
    trainer.model.load_state_dict(torch.load(run_path / "best_model_params.pt", map_location="cpu", weights_only=True))
    
    # Generate structures
    n_samples = 10000
    print(f"Generating {n_samples} structures...")
    generated_wp = trainer.generate_structures(
        n_structures=n_samples,
        calibrate=False,
        compute_validity_per_known_sequence_length=False
    )
    
    formal_validity = sum(1 for wp in generated_wp if wp is not None) / n_samples
    print(f"Formal validity: {formal_validity}")
    
    from wyckoff_transformer.evaluation.cdvae_metrics import timed_smact_validity_from_record
    from wyckoff_transformer.evaluation.statistical_evaluator import StatisticalEvaluator
    import pandas as pd
    
    # Calculate quality metrics
    valid_wp = [wp for wp in generated_wp if wp is not None]
    smact_valid_count = 0
    for wp in valid_wp:
        if timed_smact_validity_from_record(wp):
            smact_valid_count += 1
            
    smact_validity = smact_valid_count / len(valid_wp) if len(valid_wp) > 0 else 0
    
    df = pd.DataFrame(valid_wp)
    df['spacegroup_number'] = df['group']
    p1_percent = (df['spacegroup_number'] == 1).mean()
    
    import gzip
    import pickle
    data_cache_path = Path(__file__).resolve().parents[1] / "cache" / config.dataset / "data.pkl.gz"
    with gzip.open(data_cache_path, "rb") as f:
        datasets_pd = pickle.load(f)
        
    if 'structure' not in datasets_pd['test']:
        class MockStructure:
            def __len__(self): return 1
            @property
            def density(self): return 1.0
        datasets_pd['test']['structure'] = [MockStructure()] * len(datasets_pd['test'])
        
    evaluator = StatisticalEvaluator(datasets_pd['test'])
    sg_chi2 = evaluator.get_sg_chi2(df)
    elements_emd = evaluator.get_num_elements_emd(df)
    sites_emd = evaluator.get_num_sites_emd(df)
    
    print(f"SMACT validity: {smact_validity}")
    print(f"P1 percent: {p1_percent}")
    print(f"SG Chi2: {sg_chi2}")
    print(f"Elements EMD: {elements_emd}")
    print(f"Sites EMD: {sites_emd}")

    metrics = {
        "formal_validity": formal_validity,
        "smact_validity": smact_validity,
        "p1_percent": float(p1_percent),
        "sg_chi2": float(sg_chi2),
        "elements_emd": float(elements_emd),
        "sites_emd": float(sites_emd)
    }
    
    print("Metrics:", metrics)
    
    out_path = Path("/home/kna/WyckoffTransformer/src/wyckoff_transformer/tests/fixtures/ioi8tycx_reference_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
