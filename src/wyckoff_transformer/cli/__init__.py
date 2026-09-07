"""Shared argument handling for the command-line entry points."""
from typing import Dict, Optional, Sequence


def parse_condition_assignments(assignments: Sequence[str]) -> Dict[str, float]:
    """``["energy_above_hull=0", "max_force=0"]`` into ``{name: value}``.

    A model conditioned on several channels cannot be driven by one number: the CLIs used
    to fill every column of the conditioning vector with the same value, which is
    meaningless the moment the columns mean different things. Naming each one also makes
    the recorded command say what was asked for.
    """
    values: Dict[str, float] = {}
    for assignment in assignments:
        name, separator, raw = assignment.partition("=")
        if not separator or not name.strip():
            raise ValueError(
                f"--condition takes NAME=VALUE, got {assignment!r}")
        name = name.strip()
        if name in values:
            raise ValueError(f"--condition names {name!r} more than once")
        try:
            values[name] = float(raw)
        except ValueError:
            raise ValueError(
                f"--condition {name}= expects a number, got {raw!r}") from None
    return values


def resolve_condition_values(
    trainer,
    assignments: Optional[Sequence[str]],
    legacy_value: Optional[float],
) -> Optional[Dict[str, float]]:
    """What the caller asked to condition on, or None to leave it to the training distribution.

    `legacy_value` is the older single-number ``--condition-value``. It survives because
    every recorded command and script in this repo uses it, and it is unambiguous for the
    single-feature models those commands target; it is refused rather than broadcast when
    the model has more than one channel.
    """
    features = trainer.condition_features
    if assignments and legacy_value is not None:
        raise ValueError("--condition and --condition-value are alternatives; use one")
    if assignments:
        if not features:
            raise ValueError("--condition was given, but the model is not conditional.")
        return parse_condition_assignments(assignments)
    if legacy_value is None:
        return None
    if not features:
        raise ValueError("--condition-value was given, but the model is not conditional.")
    if len(features) != 1:
        raise ValueError(
            f"This model conditions on {list(features)}; --condition-value carries a "
            "single number. Pass --condition NAME=VALUE once per feature instead.")
    return {features[0]: float(legacy_value)}


def describe_condition(values: Optional[Dict[str, float]]) -> str:
    """`energy_above_hull=0, max_force=0` for a log line."""
    if not values:
        return "sampled from the training distribution"
    return ", ".join(f"{name}={value:g}" for name, value in values.items())


#: What a bare conditioning target has always meant in this repo: an energy above hull.
DEFAULT_SWEEP_FEATURE = "energy_above_hull"


def single_channel_condition(trainer, value, n_rows, feature=DEFAULT_SWEEP_FEATURE,
                             baseline=0.0):
    """`value` on one conditioning channel, `baseline` on the rest, in physical units.

    Sweeping a target means moving one variable. Writing the same number into every column
    -- which is what a `torch.full((n, condition_dim), value)` does -- moves all of them at
    once and calls the result a sweep. On a model conditioned on (energy_above_hull,
    delta_e_polymorph, max_force) a target of 0.2 would then ask for a structure 0.2 eV/atom
    above the hull *and* 0.2 above the best polymorph of its formula *and* from a relaxation
    that stopped at 0.2 eV/A, which is three changes wearing one number's clothes.

    Returns None for an unconditional model, so a caller can pass the result straight to a
    `cond=` argument. Falls back to the first channel when `feature` is not one of them,
    since the scripts that use this predate any of them having names.
    """
    features = trainer.condition_features
    if not features:
        return None
    target = feature if feature in features else features[0]
    return trainer.build_condition_from_values(
        {name: (value if name == target else baseline) for name in features}, n_rows)
