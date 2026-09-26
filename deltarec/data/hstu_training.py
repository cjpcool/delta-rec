import csv
from pathlib import Path

def load_item_ids(path):
    with Path(path).open() as f:return [int(row['item_id']) for row in csv.DictReader(f)]
