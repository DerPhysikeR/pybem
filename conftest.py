import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="plot: enable plots of integration test results to tmpdir",
    )
