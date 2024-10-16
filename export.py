import argparse
import os

from utils import apply_prompt_to_morph, load_prompt, PROMPT_DIR, Morph

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="name of morph to export", default="test")
args = parser.parse_args()

export_prompt_filepath = os.path.join(PROMPT_DIR, "export.txt")
prompt = load_prompt(export_prompt_filepath)
apply_prompt_to_morph(Morph(0, args.morph), prompt, f"export.{args.morph}")