import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-cache", action="store_true", default=False, help="run tests that require the cache file data.pkl.gz"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-cache"):
        # --run-cache given in cli: do not skip
        return
    skip_cache = pytest.mark.skip(reason="need --run-cache option to run")
    for item in items:
        if "needs_cache" in item.keywords:
            item.add_marker(skip_cache)
