#!/usr/bin/env python3
"""Standalone viewer for saved CPPN patterns.

Usage:
    python src/viewer.py outputs/paris/state_XXXX.pkl             # display
    python src/viewer.py outputs/paris/state_XXXX.pkl -o out.png  # save to file
    python src/viewer.py outputs/paris/state_XXXX.pkl -r 4096     # custom resolution
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cppn import CPPN
PATH = "/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/tangible_cppn/outputs/tests/state_2026_02_12_164907.pkl"

def main():
    parser = argparse.ArgumentParser(description="View a saved CPPN pattern")
    parser.add_argument("--pkl", default=PATH, help="Path to saved state .pkl file")
    parser.add_argument("-o", "--output", help="Save to file instead of displaying")
    parser.add_argument("-r", "--resolution", type=int, default=2048,
                        help="Render resolution (default: 2048)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip post-processing effects")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"Error: {args.pkl} not found")
        sys.exit(1)

    # Minimal params — no audio, no controller
    params = dict(
        debug=False,
        factor=16 / 9,
        reactivity="",
    )

    cppn = CPPN(output_path="/tmp", params=params)
    cppn.set_state(state_path=args.pkl)
    cppn.update_cycles_if_needed()

    # Ensure the resolution grid exists
    res = args.resolution
    if res not in cppn.coords:
        from src.cppn_utils import build_coordinate_grid
        cppn.coords[res] = build_coordinate_grid(res, cppn.factor)

    img = np.array(cppn.update(res=res))

    # Apply post-processing if saved and not skipped
    if cppn.viz_params and not args.no_postprocess:
        vp = cppn.viz_params
        cppn.grain_strength = vp.get('grain_strength', 0.0)
        cppn.displace_strength = vp.get('displace_strength', 0.0)
        cppn.invert = vp.get('invert', False)
        cppn.symmetry_mode = vp.get('symmetry_mode', 0)
        img = cppn.apply_postprocessing(img)

    # Flip vertically to match OpenGL display orientation
    img = np.flipud(img)

    im = Image.fromarray(img)
    if args.output:
        im.save(args.output)
        print(f"Saved to {args.output}")
    else:
        im.show()


if __name__ == "__main__":
    main()
