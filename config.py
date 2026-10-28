"""Loads config.json and species.json, and resolves paths relative to the project folder."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def path(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def load(config_file="config.json"):
    with open(path(config_file)) as f:
        return json.load(f)


def load_species(cfg):
    """Returns {id: species dict}. The same file feeds the website (docs/species.json)."""
    with open(path(cfg["species_file"])) as f:
        return {s["id"]: s for s in json.load(f)["species"]}
