"""The Octave mirror of the configuration must not drift from configs/default.yaml."""
import re
from pathlib import Path

import pytest

from ris_cr.config import ROOT, load_config


def _octave_params():
    txt = (ROOT / "octave" / "params.m").read_text()
    return {k: float(v) for k, v in re.findall(r"p\.(\w+)\s*=\s*([-+0-9.eE]+);", txt)}


def test_octave_params_match_yaml():
    c, o = load_config(), _octave_params()
    pairs = {"P_max_dBm": c.P_max_dBm, "B_Hz": float(c.B_Hz), "NF_dB": c.NF_dB, "C0_dB": c.C0_dB,
             "alpha_uav_ris": c.alpha["uav_ris"], "alpha_ris_ground": c.alpha["ris_ground"],
             "alpha_uav_ground": c.alpha["uav_ground"], "K_uav_ris_dB": c.K_dB["uav_ris"],
             "K_ris_ground_dB": c.K_dB["ris_ground"], "K_uav_ground_dB": c.K_dB["uav_ground"],
             "blockage_dB": c.blockage_dB, "height_m": c.uav["height_m"], "radius_m": c.uav["radius_m"]}
    for k, v in pairs.items():
        assert o[k] == pytest.approx(v), k
