# Broward Plant ID (Raspberry Pi + webcam + AlexNet)

Point a webcam at a plant and a Raspberry Pi tells you whether it is **native** to
South Florida or **invasive**, which species it most likely is, and how to
double-check by hand.

```
INVASIVE Brazilian pepper (Schinus terebinthifolia)  91%  FISC Category I
  How to check: Compound leaves with 3 to 13 leaflets on a reddish, often winged stalk. ...
  Next step: confirm the ID, then report it at eddmaps.org/florida or ask UF/IFAS Broward County Extension how to remove it safely.
  Also possible: Carrotwood 5%, Cocoplum 2%
```

Project website: https://nick-48-85.github.io/broward-plant-id/ (served from `docs/`).

## How it works

1. **Photos.** `fetch_photos.py` downloads research-grade, Creative Commons
   photos of each species from iNaturalist (Broward County first, topped up
   from the rest of Florida). `collect.py` adds your own webcam photos.
2. **Training.** `train.py` takes AlexNet pretrained on ImageNet, freezes its
   convolutional layers, and retrains the classifier on our species. Classes are
   balanced so common plants don't drown out rare ones.
3. **Identifying.** `identify.py` grabs a webcam frame on the Pi, runs it through
   the model and prints the species, native/invasive status, FISC category,
   ID tips and lookalike warnings. Guesses below `min_confidence` print UNSURE.

## The species list

`docs/species.json` holds 25 common Broward plants: 13 natives and 12 invasives
from the Florida Invasive Species Council (FISC) list. Both the model and the
website read this one file. To add a plant, add an entry (its `id` becomes the
photo folder name) and re-run `fetch_photos.py --only <id>` and `train.py`.

| Native | Invasive (FISC category) |
|---|---|
| Coontie, Sea grape, Cocoplum, Saw palmetto, Cabbage palm, Firebush, Wild coffee, American beautyberry, Gumbo limbo, Red mangrove, Dune sunflower, Boston fern, Carolina wild petunia | Brazilian pepper (I), Melaleuca (I), Australian pine (I), Old World climbing fern (I), Carrotwood (I), Air potato (I), Earleaf acacia (I), Shoebutton ardisia (I), Mexican petunia (I), Schefflera (I), Wedelia (II), Tuberous sword fern (I) |

Three pairs are deliberate lookalikes (dune sunflower / wedelia, Carolina wild
petunia / Mexican petunia, Boston fern / tuberous sword fern). The Pi prints the
difference whenever it sees one of them.

## Parts

| Part | Notes |
|---|---|
| Raspberry Pi 4 (4 GB+) or Pi 5 | 64-bit Raspberry Pi OS |
| microSD card, 32 GB+ | PyTorch is large |
| USB webcam | Autofocus (e.g. Logitech C920) helps with close-up leaves |
| Official power supply or USB-C power bank | Power bank for field use |
| A laptop/desktop | For downloading photos and training; much faster than the Pi |

## Step by step

### 1. On your laptop: get photos and train

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python fetch_photos.py --per-species 30            # quick test run first
python fetch_photos.py --other 300                 # full run: 150 per species + 300 "other" plants
python train.py --epochs 15 --site
```

* Expect 30-60 minutes for the full download (iNaturalist asks for about one
  request per second) and 10-30 minutes to train on a laptop CPU.
* `--other 300` adds random Broward plants that are not on the list, so the model
  can answer "not one of mine" instead of forcing a wrong match.
* With 100+ photos per species, try `python train.py --unfreeze-last-conv` too
  and keep whichever scores better.
* `train.py` prints species accuracy, **native vs. invasive accuracy** (the number
  that matters most) and the most common mix-ups. `--site` writes
  `docs/results.json`; commit it and the website shows the results.
* Photo credits go to `data/attributions.csv`. Keep it if you share the photos.

### 2. Set up the Pi

Flash **Raspberry Pi OS (64-bit)** onto the SD card with Raspberry Pi Imager
(https://www.raspberrypi.com/software/). In Imager's OS customisation settings,
set the hostname to `plantid`, a username and password, your Wi-Fi, and turn on
SSH. Boot the Pi with the webcam plugged in, wait 2-3 minutes, then copy this
folder over and run:

```bash
scp -r broward-plant-id pi@plantid.local:~      # from your laptop
ssh pi@plantid.local
cd ~/broward-plant-id
bash setup_pi.sh
sudo reboot
```

Copy the trained model from your laptop:

```bash
scp models/alexnet_plants.pt pi@plantid.local:~/broward-plant-id/models/
```

### 3. Identify plants

```bash
cd ~/broward-plant-id && source venv/bin/activate
python identify.py                  # press Enter for each plant
python identify.py --every 10       # automatic, every 10 seconds
python identify.py --image leaf.jpg # a saved photo
```

Every result is logged to `data/log.csv` with its photo in `data/history/`.

### 4. Improve it with your own webcam photos

Internet photos differ from what your webcam sees. 20-50 photos per species from
the Pi's own camera usually help a lot:

```bash
python collect.py --list
python collect.py sea_grape --count 30
```

Then copy `data/images/` back to the laptop and re-train.

## Settings (`config.json`)

| Setting | Meaning |
|---|---|
| `photos.place`, `photos.fallback_place` | iNaturalist places to pull photos from |
| `photos.per_species` | Photos per species to download |
| `model.min_confidence` | Below this, results say UNSURE (default 0.55) |
| `camera.device_index`, `width`, `height` | Webcam selection and resolution |

## Try it without hardware

```bash
python collect.py sea_grape --mock --count 2 --every 0.1
python identify.py --mock --once        # after training
```

## Limits

* It only knows the species in `docs/species.json` (plus "other" if trained).
* Boston fern vs. tuberous sword fern mostly differ by root tubers, which a
  camera can't see. Treat those results as "check the roots".
* It's a helper, not an authority. Confirm IDs before removing anything, and
  don't remove plants on public or protected land without permission.
  UF/IFAS Broward County Extension can confirm IDs.

## Files

| File | What it does |
|---|---|
| `docs/species.json` | Species list, shared by the model and the website |
| `docs/` | GitHub Pages website (`index.html`, `style.css`, `app.js`) |
| `config.json` / `config.py` | Settings and path helpers |
| `fetch_photos.py` | Downloads iNaturalist training photos with credits |
| `collect.py` | Takes training photos with the webcam |
| `camera.py` | Webcam capture and a fake camera for testing |
| `model.py` | AlexNet setup, image transforms, and the classifier used on the Pi |
| `train.py` | Fine-tunes AlexNet and reports accuracy |
| `identify.py` | Live identification on the Pi |
| `setup_pi.sh` | One-time Pi setup |

## Sources

* Florida Invasive Species Council plant list: https://floridainvasivespecies.org/plantlist.cfm
* UF/IFAS Assessment of Non-native Plants: https://assessment.ifas.ufl.edu/
* Institute for Regional Conservation, Natives for Your Neighborhood: https://regionalconservation.org/beta/nfyn/
* Atlas of Florida Plants: https://florida.plantatlas.usf.edu/
* Photos: iNaturalist observers, Creative Commons licenses (see `data/attributions.csv`)
