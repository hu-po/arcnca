import argparse
import os
from dataclasses import dataclass

import nbformat
from openai import OpenAI

# set up directories
root_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(root_dir, "output")
morph_dir = os.path.join(root_dir, "morphs")
os.makedirs(morph_dir, exist_ok=True)

# morph states
NOT_RUN_YET = 0
ALREADY_RAN = 0
ERRORED_OUT = -1

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: int = NOT_RUN_YET

parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="name of morph to export", default="conv")
args = parser.parse_args()

def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        return f.read()

def apply_prompt_to_morph(morph: Morph, prompt_filepath: str, new_morph_name: str) -> Morph:
    morph_nb_filepath = os.path.join(morph_dir, f"{args.morph}.ipynb")
    print(f"Reading notebook from {morph_nb_filepath}")
    with open(morph_nb_filepath, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    content = "\n\n".join(cell.source for cell in notebook.cells)
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": load_prompt(prompt_filepath)},
            {"role": "user", "content": content}
        ]
    )
    new_notebook = nbformat.v4.new_notebook()
    new_notebook.cells.append(nbformat.v4.new_code_cell(source=completion.choices[0].message.content))
    new_morph = Morph(0, new_morph_name)
    exported_nb_filepath = os.path.join(morph_dir, f"{new_morph_name}.ipynb")
    with open(exported_nb_filepath, "w", encoding="utf-8") as f:
        nbformat.write(new_notebook, f)
    print(f"New morph {new_morph.name}")
    return new_morph