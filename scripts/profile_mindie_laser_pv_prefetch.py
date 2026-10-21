#!/usr/bin/env python3
"""Original FP16 Laser vs current Vector-scale and V-only PV prefetch."""
from profile_mindie_laser_vector_scale_bulk import main


if __name__ == "__main__":
    raise SystemExit(main(default_candidate="pv_prefetch"))
