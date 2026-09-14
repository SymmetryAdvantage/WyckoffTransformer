"""WyckoffTransformer."""

# Pinned so a run's home does not depend on the shell it was launched from. Without an explicit
# entity, W&B falls back to the account default, which differs between machines and login
# sessions: run oc5zc1x4 landed under a personal entity while lu4xqw0w and t1c9ehzp went to the
# shared one, which splits a comparison across two workspaces and makes `wandb.Api().run()` on a
# two-part path fail to find runs that plainly exist. Override with --wandb-entity where a run
# genuinely belongs elsewhere.
WANDB_ENTITY = "symmetry-advantage"
WANDB_PROJECT = "WyckoffTransformer"


def wandb_run_path(run_id: str, entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT) -> str:
    """Fully qualified `entity/project/run_id` path, as `wandb.Api().run` wants it."""
    return f"{entity}/{project}/{run_id}"
