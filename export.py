"""
export a jupyter notebook for submission to kaggle (via copypaste)

pip install openai nbformat
"""

import argparse
import os
import nbformat
from openai import OpenAI

# Set up directories
root_dir = os.path.abspath(os.path.dirname(__file__))
morph_dir = os.path.join(root_dir, "morphs")
output_dir = os.path.join(root_dir, "output")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="name of morph to export", default="conv")
args = parser.parse_args()

# Read the notebook
morph_nb_filepath = os.path.join(morph_dir, f"{args.morph}.ipynb")
print(f"Reading notebook from {morph_nb_filepath}")
with open(morph_nb_filepath, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Concatenate the code cell contents
content = "\n\n".join(cell.source for cell in notebook.cells)

# Clean notebook using the updated OpenAI API
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": (
            "You are an AI that cleans Jupyter notebooks for submission to competitions. "
            "Remove all debug print statements, WandB logging, and any code related to visualizations. "
            "Do not remove or alter any essential functions, path setup, or dependencies necessary for the code to work. "
            "Make sure the output is a valid Jupyter notebook code without markdown or code block formatting, "
            "and ensure it does not include any unnecessary libraries like `arckit` if they are related to visualization."
        )},
        {"role": "user", "content": content}
    ]
)
cleaned_content = completion.choices[0].message.content

# Rebuild notebook with cleaned content
new_notebook = nbformat.v4.new_notebook()
new_notebook.cells.append(nbformat.v4.new_code_cell(source=cleaned_content.strip()))

# Save the cleaned notebook
exported_nb_filepath = os.path.join(output_dir, f"exported.{args.morph}.ipynb")
os.makedirs(output_dir, exist_ok=True)
with open(exported_nb_filepath, "w", encoding="utf-8") as f:
    nbformat.write(new_notebook, f)
print(f"Cleaned notebook saved to {exported_nb_filepath}")