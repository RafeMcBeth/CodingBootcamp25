import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from dose_calculator import calculate_dose, safety_check


def test_calculate_dose_typical():
    assert calculate_dose(1.0, 2.0) == 2.0


def test_calculate_dose_zero_time():
    assert calculate_dose(5.0, 0.0) == 0.0


def test_safety_check_over_threshold():
    msg, status = safety_check(3.0)
    assert msg == "WARNING: dose exceeds safe limit"
    assert status == "warning"


def test_safety_check_under_threshold():
    msg, status = safety_check(1.5)
    assert msg == "OK"
    assert status == "ok"
