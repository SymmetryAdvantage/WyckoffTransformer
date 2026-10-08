"""Varying logm residuals must not hide the worker's useful output."""

import warnings

from wyckoff_transformer.cryspr import relaxer


def install_capture(monkeypatch):
    seen = []
    monkeypatch.setattr(warnings, "showwarning", lambda *args: seen.append(args))
    relaxer._print_logm_warning_once()
    return seen


def emit(residual, category=RuntimeWarning):
    # Fresh registry and varying text reproduce SciPy's repeated warning.
    warnings.warn_explicit(
        f"logm result may be inaccurate, approximate err = {residual}",
        category, "/scipy/_lib/_util.py", 1138, registry={},
    )


def test_varying_residuals_print_only_the_first_warning(monkeypatch):
    seen = install_capture(monkeypatch)
    for residual in (6.718e-13, 9.013e-13, 1e-6):
        emit(residual)
    assert len(seen) == 1
    assert str(seen[0][0]).endswith("6.718e-13")
    assert seen[0][1:4] == (RuntimeWarning, "/scipy/_lib/_util.py", 1138)


def test_unrelated_warnings_and_other_categories_remain_visible(monkeypatch):
    seen = install_capture(monkeypatch)
    emit(1e-12)
    emit(2e-12, UserWarning)
    warnings.warn_explicit("other numerical warning", RuntimeWarning,
                           "/scipy/_lib/_util.py", 1138, registry={})
    assert len(seen) == 3
    assert str(seen[-1][0]) == "other numerical warning"


def test_installation_is_idempotent(monkeypatch):
    install_capture(monkeypatch)
    installed = warnings.showwarning
    relaxer._print_logm_warning_once()
    assert warnings.showwarning is installed


def test_a_forked_worker_can_print_its_own_first_warning(monkeypatch):
    seen = install_capture(monkeypatch)
    monkeypatch.setattr(relaxer.os, "getpid", lambda: 100)
    emit(1e-12)
    emit(2e-12)
    monkeypatch.setattr(relaxer.os, "getpid", lambda: 101)
    emit(3e-12)
    emit(4e-12)
    assert len(seen) == 2
