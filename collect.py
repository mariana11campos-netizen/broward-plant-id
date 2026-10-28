"""Take your own training photos with the Pi's webcam.

iNaturalist photos are taken with phones in all sorts of light. Adding 20-50
photos per species from *your* webcam teaches the model what your camera sees,
which usually helps more than another 100 internet photos.

    python collect.py --list                     # show species ids
    python collect.py sea_grape --count 30       # press Enter between shots
    python collect.py wedelia --count 30 --every 3   # automatic, one shot every 3 s

Move around between shots: different leaves, angles, distances and light.
Photos go to data/images/<species_id>/cam_*.jpg.
"""
import argparse
import os
import time
from datetime import datetime

from PIL import Image

import camera
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="?", help="species id from docs/species.json, or 'other'")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--every", type=float, default=0, help="seconds between shots; 0 = wait for Enter")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mock", action="store_true", help="fake camera, for testing")
    args = ap.parse_args()
    cfg = config.load()
    species = config.load_species(cfg)

    if args.list or not args.species:
        for sid, s in species.items():
            print(f"{sid:26s} {s['status']:8s} {s['common']}")
        return
    if args.species not in species and args.species != "other":
        raise SystemExit(f"Unknown species {args.species!r}. Run with --list to see ids.")

    cam = camera.MockWebcam(cfg) if args.mock else camera.Webcam(cfg)
    folder = config.path(f"data/images/{args.species}")
    os.makedirs(folder, exist_ok=True)
    try:
        for i in range(args.count):
            if args.every:
                time.sleep(args.every if i else 0)
            else:
                input(f"[{i + 1}/{args.count}] Point the camera at the plant, then press Enter...")
            name = "cam_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            Image.fromarray(cam.capture()).save(os.path.join(folder, name), quality=92)
            print(f"  saved {args.species}/{name}")
    finally:
        cam.close()
    print("Done. Next: python train.py")


if __name__ == "__main__":
    main()
