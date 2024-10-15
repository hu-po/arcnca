import os
from dataclasses import dataclass

import nbformat
from openai import OpenAI

# set up directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MORPH_DIR = os.path.join(ROOT_DIR, "morphs")
PROMPT_DIR = os.path.join(ROOT_DIR, "prompts")
os.makedirs(MORPH_DIR, exist_ok=True)

# morph states
NOT_RUN_YET = 0
ALREADY_RAN = -1
ERRORED_OUT = -2

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: int = NOT_RUN_YET

def load_prompt(prompt_path):
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def apply_prompt_to_morph(morph: Morph, prompt: str, new_morph_name: str) -> Morph:
    morph_nb_filepath = os.path.join(MORPH_DIR, f"{morph.name}.ipynb")
    print(f"Reading notebook from {morph_nb_filepath}")
    with open(morph_nb_filepath, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    content = "\n\n".join(cell.source for cell in notebook.cells)
    format_prompt_filepath = os.path.join(PROMPT_DIR, "format.txt")
    prompt += load_prompt(format_prompt_filepath)
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    )
    reply = completion.choices[0].message.content
    new_notebook = nbformat.v4.new_notebook()
    new_notebook.cells.append(nbformat.v4.new_code_cell(source=reply))
    new_morph = Morph(0, new_morph_name)
    exported_nb_filepath = os.path.join(MORPH_DIR, f"{new_morph_name}.ipynb")
    with open(exported_nb_filepath, "w", encoding="utf-8") as f:
        nbformat.write(new_notebook, f)
    print(f"New morph {new_morph.name}")
    return new_morph