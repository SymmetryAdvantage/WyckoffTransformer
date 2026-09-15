"""Answer, for the PBS chain, whether a run has resume state anywhere.

`scripts/platforms/aspire2a/train_in_pbs.sh` has to decide between resuming a pinned W&B run id and
minting a fresh one, and it must not get that wrong in the direction of starting
over: a run whose `last_checkpoint.pt` was purged from `runs/` has trained epochs
that exist only in its W&B artifact.

Exit status:
    0  resume state is available -- locally, or as a W&B artifact to download
    1  there is genuinely nothing to resume; the id may be re-minted
    2  the question could not be answered (offline, no credentials, API error)

Status 2 is deliberately distinct: the caller must not read "cannot tell" as
"nothing to resume".
"""

import argparse
import logging
import sys
from pathlib import Path

from wyckoff_transformer import wandb_run_path
from wyckoff_transformer.paths import runs_root
from wyckoff_transformer.trainer import CHECKPOINT_FILENAME

AVAILABLE, NOTHING_TO_RESUME, UNKNOWN = 0, 1, 2

logger = logging.getLogger("resume_probe")


def local_checkpoint(run_id: str, runs: Path | None = None) -> Path | None:
    """The run's checkpoint on this machine, if it is there."""
    root = runs_root() if runs is None else Path(runs)
    candidate = root / run_id / CHECKPOINT_FILENAME
    return candidate if candidate.is_file() else None


#: How the API reports a run that was never logged to, as opposed to one it could not
#: reach. `CommError` covers both, so the message is the only signal there is; an
#: unrecognised failure stays `None` rather than being read as "no checkpoint".
_NOT_FOUND_MARKERS = ("could not find run", "not found")


def wandb_checkpoint_exists(run_id: str, entity: str, project: str) -> bool | None:
    """Whether W&B holds a mirrored resume checkpoint for *run_id*.

    The trainer mirrors it as one of the run's files rather than an artifact, so this
    asks for that file and reads the absence off its size -- a file a run never
    uploaded comes back as a zero-sized entry, not an error.

    Returns ``True`` if it is there, ``False`` if the run definitively has none --
    including a run id W&B has never seen, the ordinary case for an id minted locally
    by a link that crashed before logging -- and ``None`` when the question could not
    be answered at all.
    """
    try:
        import wandb

        api = wandb.Api()
        run = api.run(wandb_run_path(run_id, entity, project))
    except Exception as exc:  # noqa: BLE001 - offline, no key, deleted run, API outage
        message = str(exc).lower()
        if all(marker in message for marker in _NOT_FOUND_MARKERS):
            logger.info("W&B has never seen run %s", run_id)
            return False
        logger.warning("Cannot ask W&B about %s: %s", run_id, exc)
        return None
    try:
        remote = run.file(CHECKPOINT_FILENAME)
    except Exception as exc:  # noqa: BLE001 - the run resolved but its files did not
        logger.warning("Cannot list the files of %s: %s", run_id, exc)
        return None
    return getattr(remote, "size", 0) not in (0, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--runs-path", type=Path, default=None,
                        help="Where run directories live (default: the runs store).")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    found = local_checkpoint(args.run_id, args.runs_path)
    if found is not None:
        print(f"local {found}")
        return AVAILABLE

    from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT

    in_wandb = wandb_checkpoint_exists(
        args.run_id, args.entity or WANDB_ENTITY, args.project or WANDB_PROJECT)
    if in_wandb is None:
        print("unknown")
        return UNKNOWN
    if in_wandb:
        print(f"wandb {args.run_id}:{CHECKPOINT_FILENAME}")
        return AVAILABLE
    print("none")
    return NOTHING_TO_RESUME


if __name__ == "__main__":
    sys.exit(main())
