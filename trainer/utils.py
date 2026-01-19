import json
import csv
import os
from datetime import datetime


def create_experiment_dir(base_dir, model_name):
    exp_dir = os.path.join(base_dir, model_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_csv(history, path):
    keys = history[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
