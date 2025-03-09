import json
import os

CONFIG_FILE = "config.json"

def load_config():
    """Loads config from JSON file if it exists."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "rt", encoding="utf-8") as f:
            return json.load(f)
    return None  # Return None if no config exists

def save_config(config):
    """Saves a dictionary of settings to a JSON file."""
    with open(CONFIG_FILE, "wt", encoding="utf-8") as f:
        json.dump(config, f, indent=4)