#!/usr/bin/env python3
"""Original FP16 Laser vs V-only PV prefetch and added same-descriptor P prefetch."""
from profile_mindie_laser_vector_scale_bulk import main


if __name__ == "__main__":
    raise SystemExit(main(default_candidate="pv_p_prefetch"))
