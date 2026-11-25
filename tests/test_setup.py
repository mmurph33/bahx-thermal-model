"""Environment checks: every src module imports and CoolProp works.

These run before any physics is implemented, so a fresh clone can confirm
that the virtual environment, pytest.ini and the package layout are right.
"""

import importlib

import pytest
from CoolProp.CoolProp import PropsSI

SRC_MODULES = [
    "src.config",
    "src.properties",
    "src.correlations",
    "src.steady",
    "src.transient",
    "src.metrics",
    "src.stress",
]


@pytest.mark.parametrize("name", SRC_MODULES)
def test_module_imports(name):
    """Each src module imports cleanly (catches missing packages and import cycles)."""
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} should have a module docstring"


def test_coolprop_methane_cp():
    """CoolProp gives a sensible methane heat capacity at 300 K and 1 atm.

    Reference: NIST Chemistry WebBook, methane at 300 K and 0.101325 MPa,
    cp = 2.2359 J/(g K) = 2235.9 J/(kg K). A 1 % band is "sensible" without
    depending on the CoolProp version.
    """
    cp = PropsSI("CPMASS", "T", 300.0, "P", 101325.0, "Methane")
    assert cp == pytest.approx(2235.9, rel=0.01)
