#!/usr/bin/env python3
"""Calibrate current Vector-scale Cube probes alongside original FP16 Laser."""
from profile_mindie_laser_cube_phases import main as cube_main


def main(argv=None):
    return cube_main(argv, default_families=("vscale", "fp16"),
                     default_output="results/mindie_laser_vscale_phases.json")


if __name__ == "__main__":
    raise SystemExit(main())
