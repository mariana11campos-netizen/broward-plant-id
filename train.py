"""Fine-tune AlexNet on photos in data/images/<species_id>/.

Fill the folders first with fetch_photos.py (iNaturalist) and/or collect.py
(your own webcam). Folder names must match ids in docs/species.json; an extra
folder named "other" (random plants that are not on the list) is allowed and
lets the model say "not one of mine".

Training on a laptop or desktop is much faster than on the Pi. Copy
models/alexnet_plants.pt to the Pi afterwards.

    python train.py
    python train.py --epochs 20 --unfreeze-last-conv
    python train.py --site      # also write docs/results.json for the website
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

import config
from model import build_alexnet, eval_tf, train_tf

MIN_PER_CLASS = 10


def split_indices(targets, val_frac, seed=0):
    """Per-class split so every species shows up in validation."""
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for c in sorted(set(targets)):
        idx = [i for i, t in enumerate(targets) if t == c]
        idx = [idx[i] for i in torch.randperm(len(idx), generator=g)]
        n_val = max(1, round(len(idx) * val_frac))
        val_idx += idx[:n_val]
        train_idx += idx[n_val:]
    return train_idx, val_idx


def evaluate(net, dl, n_classes):
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    net.eval()
    with torch.no_grad():
        for x, y in dl:
            for t, p in zip(y, net(x).argmax(1)):
                confusion[t, p] += 1
    return confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.2, help="fraction of photos held out for validation")
    ap.add_argument("--unfreeze-last-conv", action="store_true",
                    help="also retrain AlexNet's last conv layer (better with 100+ photos per species)")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (testing only)")
    ap.add_argument("--site", action="store_true", help="write docs/results.json for the website")
    args = ap.parse_args()
    cfg = config.load()
    species = config.load_species(cfg)

    root = config.path("data/images")
    if not os.path.isdir(root):
        raise SystemExit("No data/images/ folder. Run fetch_photos.py or collect.py first.")
    folders = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    unknown = [d for d in folders if d not in species and d != "other"]
    if unknown:
        raise SystemExit(f"Folders not in docs/species.json: {unknown}. Rename them or add the species.")

    train_set = datasets.ImageFolder(root, transform=train_tf)
    val_set = datasets.ImageFolder(root, transform=eval_tf)
    classes = train_set.classes
    counts = Counter(train_set.targets)
    small = [classes[c] for c, n in counts.items() if n < MIN_PER_CLASS]
    if small:
        raise SystemExit(f"Need at least {MIN_PER_CLASS} photos for: {small}")
    if len(classes) < 2:
        raise SystemExit("Need at least 2 species folders.")
    missing = [s for s in species if s not in classes]
    if missing:
        print(f"Note: no photos yet for {missing}; the model won't know them.")

    train_idx, val_idx = split_indices(train_set.targets, args.val)
    # Balance the classes: iNaturalist has far more Brazilian pepper photos than coontie photos.
    weights = [1.0 / counts[train_set.targets[i]] for i in train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
    train_dl = DataLoader(Subset(train_set, train_idx), batch_size=args.batch, sampler=sampler, num_workers=2)
    val_dl = DataLoader(Subset(val_set, val_idx), batch_size=args.batch, num_workers=2)
    print(f"{len(classes)} classes, {len(train_idx)} training / {len(val_idx)} validation photos")

    net = build_alexnet(len(classes), pretrained=not args.no_pretrained,
                        unfreeze_last_conv=args.unfreeze_last_conv)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state, best_conf = -1.0, None, None
    for ep in range(args.epochs):
        net.train()
        total_loss = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
        conf = evaluate(net, val_dl, len(classes))
        acc = conf.diag().sum().item() / conf.sum().item()
        print(f"epoch {ep + 1}/{args.epochs}  loss {total_loss / len(train_idx):.3f}  val acc {acc:.0%}")
        if acc > best_acc:
            best_acc, best_conf = acc, conf
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    # Native vs. invasive accuracy matters more than exact species: calling a native
    # sea grape "cocoplum" is harmless, calling it "Brazilian pepper" is not.
    def status(c):
        return species[c]["status"] if c in species else "other"
    group_ok = sum(best_conf[t, p].item() for t in range(len(classes)) for p in range(len(classes))
                   if status(classes[t]) == status(classes[p]))
    per_class = {c: round(best_conf[i, i].item() / max(1, best_conf[i].sum().item()), 3)
                 for i, c in enumerate(classes)}
    mixups = sorted(((best_conf[t, p].item(), classes[t], classes[p])
                     for t in range(len(classes)) for p in range(len(classes))
                     if t != p and best_conf[t, p] > 0), reverse=True)[:8]
    results = {
        "trained": datetime.now().isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "photos": {c: counts[i] for i, c in enumerate(classes)},
        "val_photos": len(val_idx),
        "species_accuracy": round(best_acc, 3),
        "native_vs_invasive_accuracy": round(group_ok / best_conf.sum().item(), 3),
        "per_class_accuracy": per_class,
        "top_mixups": [{"actual": a, "predicted": p, "count": n} for n, a, p in mixups],
    }
    print(f"Best: species accuracy {best_acc:.0%}, "
          f"native vs. invasive accuracy {results['native_vs_invasive_accuracy']:.0%}")
    for m in results["top_mixups"][:5]:
        print(f"  mixed up {m['actual']} -> {m['predicted']} ({m['count']}x)")

    out = config.path(cfg["model"]["weights_path"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes}, out)
    with open(config.path("models/results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out} and models/results.json")
    if args.site:
        with open(config.path("docs/results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("Wrote docs/results.json (commit it to show results on the website)")


if __name__ == "__main__":
    main()
