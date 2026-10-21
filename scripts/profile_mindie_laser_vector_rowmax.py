#!/usr/bin/env python3
"""Original FP16 Laser vs Vector-scale baseline and isolated row-max candidate."""
from profile_mindie_laser_vector_scale_bulk import main


if __name__ == "__main__":
    raise SystemExit(main(default_candidate="rowmax"))
