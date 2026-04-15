import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-cache", action="store_true", default=False, help="run tests that require the cache file data.pkl.gz"
    )
    parser.addoption(
        "--run-relax", action="store_true", default=False,
        help="run tests that require network access and a MACE model (needs_relax marker)",
    )

def pytest_collection_modifyitems(config, items):
    skip_cache = pytest.mark.skip(reason="need --run-cache option to run")
    skip_relax = pytest.mark.skip(reason="need --run-relax option to run")
    for item in items:
        if "needs_cache" in item.keywords and not config.getoption("--run-cache"):
            item.add_marker(skip_cache)
        if "needs_relax" in item.keywords and not config.getoption("--run-relax"):
            item.add_marker(skip_relax)
