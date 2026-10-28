"""Point the webcam at a plant and find out if it's native or invasive.

    python identify.py                 # press Enter to identify, Ctrl+C to quit
    python identify.py --every 10      # identify every 10 seconds
    python identify.py --image leaf.jpg    # identify a saved photo (works on a laptop)
    python identify.py --mock --once   # software test with a fake camera

Each result is appended to data/log.csv and its photo kept in data/history/.
"""
import argparse
import csv
import os
import time
from datetime import datetime

from PIL import Image

import camera
import config
from model import PlantClassifier

LOG_FIELDS = ["time", "image", "species", "status", "confidence", "verdict", "runner_up", "runner_up_conf"]


def describe(preds, species, min_conf):
    """Turns model output into the text shown to the user, plus a short verdict for the log."""
    (best, conf), rest = preds[0], preds[1:]
    sp = species.get(best)
    lines = []
    if best == "other":
        verdict = "OTHER"
        lines.append(f"OTHER ({conf:.0%})  Doesn't look like any plant on the list.")
    elif conf < min_conf:
        verdict = "UNSURE"
        lines.append(f"UNSURE  Best guess {sp['common']} ({sp['status']}, {conf:.0%}). "
                     "Get closer, fill the frame with leaves, or try better light.")
    else:
        verdict = sp["status"].upper()
        fisc = f"  FISC Category {sp['fisc']}" if sp.get("fisc") else ""
        lines.append(f"{verdict:8s} {sp['common']} ({sp['scientific']})  {conf:.0%}{fisc}")
        lines.append(f"  How to check: {sp['id_tips']}")
        if sp["status"] == "invasive":
            lines.append("  Next step: confirm the ID, then report it at eddmaps.org/florida or ask "
                         "UF/IFAS Broward County Extension how to remove it safely.")
    if sp and sp.get("lookalike") and sp["lookalike"] in species:
        other = species[sp["lookalike"]]
        lines.append(f"  Lookalike: {other['common']} ({other['status']}). {other['id_tips']}")
    if rest:
        lines.append("  Also possible: " + ", ".join(
            f"{species[c]['common'] if c in species else c} {p:.0%}" for c, p in rest))
    return verdict, "\n".join(lines)


def log(cfg, img, preds, species, verdict):
    hist = config.path("data/history")
    os.makedirs(hist, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    img.save(os.path.join(hist, name), quality=90)
    path = config.path(cfg["log_path"])
    new = not os.path.exists(path)
    (best, conf), second = preds[0], (preds[1] if len(preds) > 1 else ("", 0.0))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "image": name,
                    "species": best, "status": species.get(best, {}).get("status", "other"),
                    "confidence": round(conf, 3), "verdict": verdict,
                    "runner_up": second[0], "runner_up_conf": round(second[1], 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="identify this photo instead of using the webcam")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=0, help="seconds between checks; 0 = wait for Enter")
    ap.add_argument("--mock", action="store_true", help="fake camera, for testing")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    cfg = config.load()
    species = config.load_species(cfg)
    weights = config.path(cfg["model"]["weights_path"])
    if not os.path.exists(weights):
        raise SystemExit(f"No model at {weights}. Run train.py first (or copy the .pt file here).")
    clf = PlantClassifier(weights)
    mc = cfg["model"]

    def run(img):
        preds = clf.predict(img, mc["top_k"])
        verdict, text = describe(preds, species, mc["min_confidence"])
        print(text, flush=True)
        if not args.no_log:
            log(cfg, img, preds, species, verdict)

    if args.image:
        return run(Image.open(args.image))

    cam = camera.MockWebcam(cfg) if args.mock else camera.Webcam(cfg)
    try:
        while True:
            if not args.once:
                if args.every:
                    time.sleep(args.every)
                else:
                    input("\nPoint the camera at a plant and press Enter (Ctrl+C to quit)...")
            run(Image.fromarray(cam.capture()))
            if args.once:
                break
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()


if __name__ == "__main__":
    main()
