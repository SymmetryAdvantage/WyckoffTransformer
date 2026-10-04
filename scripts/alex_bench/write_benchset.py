"""Write the DiffCSP++ bench-harness input for a protocol run's unique genes.

DiffCSPNew's ``bench/gen_init.py`` reads a pickled list of ``{"rep": gene}``; every
prediction it makes carries the list position as ``entry``. Each entry here also carries
``gene_index``, the gene's position in the protocol's gene file, so that
``export_predictions.py`` can file each structure under the gene it was made for. Only the
screen's representatives are written when ``--protocol-dir`` names a protocol run: a
duplicate gene is the same reconstruction task. Without it every gene is written, which is
right for a gene file a rule of engagement has already screened for uniqueness.

    .venv/bin/python scripts/alex_bench/write_benchset.py GENES.json.gz OUT.pkl \
        [--protocol-dir PROTOCOL_DIR]
"""
import argparse
import pickle
from pathlib import Path

from wyckoff_transformer.cli.protocol import SCREEN_FILE, _todo_representatives
from wyckoff_transformer.evaluation.protocol import load_genes, read_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("genes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--protocol-dir", type=Path, default=None,
                        help="Where --stage screen wrote screen.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    genes = load_genes(args.genes)
    if args.protocol_dir is not None:
        todo = _todo_representatives(read_screen(args.protocol_dir / SCREEN_FILE), args.limit)
    else:
        todo = list(range(len(genes) if args.limit is None else min(args.limit, len(genes))))
    entries = [{"rep": genes[index], "gene_index": int(index)} for index in todo]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(entries, handle)
    print(f"{len(entries)} unique genes of {len(genes)} -> {args.out}")


if __name__ == "__main__":
    main()
