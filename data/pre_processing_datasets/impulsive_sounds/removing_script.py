from utils import RemovingNonImpulsive
import os 

from config import datasets_to_clean

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ESC-50v2")

args = parser.parse_args()
args = vars(args)
dataset = args["dataset"]

src_root_dir = datasets_to_clean[dataset]["src_dir"]
dst_root_dir = datasets_to_clean[dataset]["dst_dir"]

i = RemovingNonImpulsive(original_root_dir=src_root_dir,
                         only_impulse_root_dir=dst_root_dir)

i.generate_only_impulse_dataset()

