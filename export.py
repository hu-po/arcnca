"""
export a jupyter notebook for submission to kaggle (via copypaste)

pip install openai nbformat
"""

import argparse

from utils import apply_prompt_to_morph

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="name of morph to export", default="conv")
args = parser.parse_args()

apply_prompt_to_morph(args.morph, "prompts/export.txt", f"export.{args.morph}")