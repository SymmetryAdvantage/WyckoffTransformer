"""Selection and error statistics for one observed structure per Wyckoff gene."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_minima(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select by observed intensive energy before prediction or novelty filtering.

    Each pool is an explicitly declared set of comparable observations. A pool
    contains one energy source and, for protocol runs, one relaxation track.
    Ties use the saved geometry identifier, then the immutable candidate ID.
    Rows without a final gene or finite energy remain in the selection audit.
    Missing predictions never cause substitution of a higher-energy structure.
    """
    if frame.candidate_id.duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    keyed = frame.group_key.notna() & (frame.group_key.fillna("") != "")
    finite = np.isfinite(frame.truth)
    eligible = frame[keyed & finite].copy()
    eligible["_geometry_tie"] = eligible.geometry.fillna("").astype(str)
    ordered = eligible.sort_values(["truth", "_geometry_tie", "candidate_id"], kind="stable")
    selected = ordered.drop_duplicates(["pool", "group_key"]).drop(columns="_geometry_tie")
    audit = frame[["candidate_id", "pool", "group_key", "truth"]].copy()
    audit["selection_status"] = np.select(
        [~keyed, ~finite, frame.candidate_id.isin(selected.candidate_id)],
        ["unresolved gene", "missing energy", "selected"],
        default="higher energy or tie",
    )
    expected = eligible.groupby(["pool", "group_key"]).truth.min().sort_index()
    actual = selected.set_index(["pool", "group_key"]).truth.sort_index()
    if not expected.equals(actual):
        raise ValueError("Selection is not the unique observed minimum of every pool/gene")
    return selected.sort_values(["pool", "group_key"]), audit


def error_statistics(frame: pd.DataFrame, prediction: str = "final_prediction") -> dict:
    """Report finite pairs and keep unsupported predictions in the denominator."""
    p = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame.truth, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(p) & np.isfinite(y)
    p, y = p[keep], y[keep]
    r = p - y
    result = dict(n_rows=len(frame), n=int(keep.sum()), excluded=int((~keep).sum()))
    if len(r):
        result.update(
            mae=float(abs(r).mean()), rmse=float(np.sqrt((r * r).mean())),
            bias=float(r.mean()), median_absolute_error=float(np.median(abs(r))),
            p90_absolute_error=float(np.quantile(abs(r), .9)),
            within_50meV=float((abs(r) <= .05).mean()),
            within_100meV=float((abs(r) <= .1).mean()),
        )
        if len(r) > 1 and np.std(p) > 0 and np.std(y) > 0:
            result["pearson"] = float(np.corrcoef(p, y)[0, 1])
            result["spearman"] = float(pd.Series(p).corr(pd.Series(y), method="spearman"))
    return result


def add_archive_candidates(generated: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Offer the archive minimum for each gene with a finite generated target.

    The union has the same eligible genes as the generated-only pool. It does
    not add archive-only genes or replace unsupported model predictions.
    Novelty against LeMat-fixed is known for archive entries; MP-20 novelty is
    not inferred. The chosen reference ID is retained for reconstruction.
    """
    eligible = generated[generated.final_key.notna() & np.isfinite(generated.truth)]
    predictions = eligible.groupby("final_key").final_prediction
    if (predictions.nunique() > 1).any():
        raise ValueError("One final gene has inconsistent saved predictions")
    keys = set(eligible.final_key)
    candidates = reference[reference.key.isin(keys)].sort_values(
        ["formation_energy_per_atom", "immutable_id"], kind="stable",
    ).drop_duplicates("key")
    if candidates.empty:
        return generated.copy()
    source = eligible.drop_duplicates("final_key").set_index("final_key")
    archive = source.loc[candidates.key].copy().reset_index()
    archive["truth"] = candidates.formation_energy_per_atom.to_numpy()
    archive["gene_min_formation_energy_per_atom"] = archive.truth
    archive["cohort"] = "LeMat-Bulk-fixed archive"
    archive["material_id"] = candidates.immutable_id.to_numpy()
    archive["geometry"] = "archive:" + candidates.immutable_id.to_numpy()
    archive["candidate_id"] = "archive:" + candidates.immutable_id.to_numpy()
    archive["origin"] = "archive"
    archive["novel_structure_fixed"] = False
    archive["novel_structure_mp20"] = np.nan
    archive["final_gene_novel_fixed"] = False
    archive["valid_structure"] = True
    archive["reference_split"] = candidates.split.to_numpy()
    # Geometry, total energy and other per-structure fields inherited from the
    # generated row would misdescribe the archive entry. Keep only named fields.
    keep = [
        "candidate_id", "dataset", "cohort", "material_id", "geometry", "origin",
        "final_key", "final_prediction", "truth", "gene_min_formation_energy_per_atom",
        "gene_in_train", "n_train", "n_val", "n_test", "n_reference_rows",
        "final_gene_novel_fixed", "novel_structure_fixed", "novel_structure_mp20",
        "valid_structure", "reference_split",
    ]
    archive = archive[[c for c in keep if c in archive]]
    return pd.concat([generated, archive], ignore_index=True)
