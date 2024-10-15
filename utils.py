import argparse
import os
from dataclasses import dataclass
import subprocess
import psutil

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
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def apply_prompt_to_morph(morph: Morph, prompt_filepath: str, new_morph_name: str) -> Morph:
    morph_nb_filepath = os.path.join(MORPH_DIR, f"{args.morph}.ipynb")
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
    exported_nb_filepath = os.path.join(MORPH_DIR, f"{new_morph_name}.ipynb")
    with open(exported_nb_filepath, "w", encoding="utf-8") as f:
        nbformat.write(new_notebook, f)
    print(f"New morph {new_morph.name}")
    return new_morph

def get_device_memory():
    used_mem_mb = None
    total_mem_mb = None

    # Try to get memory info using nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        total_mem = used_mem = 0
        for line in result.stdout.strip().split('\n'):
            total, used = map(float, line.strip().split(', '))
            total_mem += total
            used_mem += used
        return used_mem, total_mem
    except Exception:
        pass  # nvidia-smi not available

    # Try to get memory info from JAX devices
    try:
        import jax
        devices = jax.devices()
        if devices:
            total_mem = sum(device.memory_size() for device in devices)
            total_mem_mb = total_mem / (1024 * 1024)  # Bytes to MB
            # Used memory is not readily available via JAX
            print(f"Total device memory: {total_mem_mb:.2f} MB")
            return None, total_mem_mb
    except Exception:
        pass  # JAX not available or no devices found

    # Try to get memory info using tegrastats (for AGX Orin devices)
    try:
        result = subprocess.run(
            ['tegrastats', '--interval', '1', '--count', '1'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        # Parse tegrastats output if necessary
        print("Tegrastats output:", output)
        return None, None
    except Exception:
        pass  # tegrastats not available

    # Fallback to system memory info using psutil
    try:
        mem = psutil.virtual_memory()
        total_mem_mb = mem.total / (1024 * 1024)
        used_mem_mb = mem.used / (1024 * 1024)
        return used_mem_mb, total_mem_mb
    except Exception:
        pass  # psutil not available

    print("Could not retrieve device memory usage.")
    return None, None
