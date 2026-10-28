"""Download training photos for every species from iNaturalist.

Uses research-grade observations (identified by at least two people) with a
Creative Commons or public-domain photo license, from Broward County first and
the rest of Florida to top up species with too few local photos. Every photo's
author and license is written to data/attributions.csv, which you must keep if
you share the photos.

    python fetch_photos.py                    # all species, per_species from config.json
    python fetch_photos.py --per-species 50   # quicker test run
    python fetch_photos.py --only coontie wedelia
    python fetch_photos.py --other 300        # random local plants for the "other" class

Run it on a laptop with a good connection (it's a few thousand small images),
then copy data/images/ to the Pi or train on the laptop.
"""
import argparse
import csv
import os
import time

import requests

import config

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "broward-plant-id/1.0 (student project; AlexNet plant classifier)"}
PAUSE = 1.0  # iNaturalist asks for about one API request per second
ATTRIB_FIELDS = ["file", "species", "observation", "attribution", "license"]


class INat:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def get(self, endpoint, **params):
        for attempt in range(4):
            r = self.s.get(f"{API}/{endpoint}", params=params, timeout=30)
            time.sleep(PAUSE)
            if r.status_code == 429:  # rate limited: back off
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("iNaturalist kept rate-limiting; try again later.")

    def place_id(self, name):
        for p in self.get("places/autocomplete", q=name)["results"]:
            if p["display_name"].lower().startswith(name.lower()):
                return p["id"], p["display_name"]
        raise SystemExit(f"iNaturalist has no place called {name!r}.")

    def taxon_id(self, scientific):
        # "Heptapleurum actinophyllum (Schefflera actinophylla)" -> try both names.
        names = [n.strip(" )") for n in scientific.split("(")]
        for name in names:
            for t in self.get("taxa", q=name, rank="species,subspecies,variety")["results"]:
                if t["name"].lower() == name.lower() or name.lower() in [
                        s.lower() for s in t.get("names", [])]:
                    return t["id"]
            res = self.get("taxa", q=name)["results"]
            if res and res[0].get("matched_term", "").lower() == name.lower():
                return res[0]["id"]
        raise SystemExit(f"iNaturalist doesn't know {scientific!r}. Fix the name in docs/species.json.")

    def observations(self, licenses, **filters):
        """Yields research-grade observations, newest first, until the caller stops."""
        id_below = None
        while True:
            params = dict(quality_grade="research", photos="true", photo_license=",".join(licenses),
                          per_page=200, order_by="id", order="desc", **filters)
            if id_below:
                params["id_below"] = id_below
            results = self.get("observations", **params)["results"]
            if not results:
                return
            yield from results
            id_below = results[-1]["id"]


def save_photos(inat, observations, folder, label, want, licenses, attrib, seen):
    os.makedirs(folder, exist_ok=True)
    have = len([f for f in os.listdir(folder) if f.startswith("inat_")])
    for obs in observations:
        if have >= want:
            break
        photo = next((p for p in obs["photos"] if p.get("license_code") in licenses), None)
        if not photo or photo["id"] in seen:
            continue
        name = f"inat_{photo['id']}.jpg"
        url = photo["url"].replace("/square.", "/medium.")  # 500 px on the long side
        r = inat.s.get(url, timeout=30)
        if r.status_code != 200:
            continue
        with open(os.path.join(folder, name), "wb") as f:
            f.write(r.content)
        seen.add(photo["id"])
        attrib.writerow({"file": f"{label}/{name}", "species": label,
                         "observation": f"https://www.inaturalist.org/observations/{obs['id']}",
                         "attribution": photo.get("attribution", ""), "license": photo["license_code"]})
        have += 1
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int)
    ap.add_argument("--only", nargs="+", help="species ids to fetch (default: all)")
    ap.add_argument("--other", type=int, default=0,
                    help="also fetch this many photos of random Broward plants NOT on the list")
    args = ap.parse_args()
    cfg = config.load()
    pc = cfg["photos"]
    want = args.per_species or pc["per_species"]
    licenses = pc["licenses"]
    species = config.load_species(cfg)
    ids = args.only or list(species)
    bad = [i for i in ids if i not in species]
    if bad:
        raise SystemExit(f"Unknown species ids: {bad}")

    inat = INat()
    local_id, local_name = inat.place_id(pc["place"])
    wide_id, wide_name = inat.place_id(pc["fallback_place"])
    print(f"Photos from {local_name}, topped up from {wide_name}")

    root = config.path("data/images")
    attrib_path = config.path("data/attributions.csv")
    seen = set()
    if os.path.exists(attrib_path):
        with open(attrib_path) as f:
            seen = {int(r["file"].split("inat_")[1].split(".")[0]) for r in csv.DictReader(f)
                    if "inat_" in r["file"]}
    new_file = not os.path.exists(attrib_path)
    os.makedirs(root, exist_ok=True)
    with open(attrib_path, "a", newline="") as f:
        attrib = csv.DictWriter(f, fieldnames=ATTRIB_FIELDS)
        if new_file:
            attrib.writeheader()
        for sid in ids:
            sp = species[sid]
            taxon = inat.taxon_id(sp["scientific"])
            folder = os.path.join(root, sid)
            n = save_photos(inat, inat.observations(licenses, taxon_id=taxon, place_id=local_id),
                            folder, sid, want, licenses, attrib, seen)
            local = n
            if n < want:
                n = save_photos(inat, inat.observations(licenses, taxon_id=taxon, place_id=wide_id),
                                folder, sid, want, licenses, attrib, seen)
            f.flush()
            print(f"  {sid:26s} {n:4d} photos ({local} from {pc['place']})")

        if args.other:
            # Plants observed in Broward that are not on our list, so the model can say "other".
            taxa = [inat.taxon_id(species[s]["scientific"]) for s in species]
            obs = inat.observations(licenses, place_id=local_id, iconic_taxa="Plantae",
                                    without_taxon_id=",".join(map(str, taxa)))
            n = save_photos(inat, obs, os.path.join(root, "other"), "other", args.other,
                            licenses, attrib, seen)
            print(f"  {'other':26s} {n:4d} photos")
    print("Done. Credits are in data/attributions.csv. Next: python train.py")


if __name__ == "__main__":
    main()
