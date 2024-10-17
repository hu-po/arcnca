import argparse
from dataclasses import dataclass
import os
import random
import subprocess
import time
import uuid
import yaml
from typing import List

import arxiv
import nbformat
from openai import OpenAI

# set up directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MORPH_DIR = os.path.join(ROOT_DIR, "morphs")
PROMPT_DIR = os.path.join(ROOT_DIR, "prompts")
MUTATION_PROMPTS_DIR = os.path.join(PROMPT_DIR, "mutations")
os.makedirs(MORPH_DIR, exist_ok=True)

# choose which mutations are active
MUTATIONS: List[str] = [
    "open_ended",
    "rewrite_model_and_train",
    "rewrite_model",
    "tune_config",
]

# morph states
NOT_RUN_YET = 0
ALREADY_RAN = -1
ERRORED_OUT = -2

@dataclass(order=True)
class Morph:
    score: float
    name: str
    state: int = NOT_RUN_YET

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="gpt")
parser.add_argument("--tb", action="store_true", help="start tensorboard session")
parser.add_argument("--protomorphs", type=str, default="base", help="comma separated list of protomorphs to seed evolution")
parser.add_argument("--num_rounds", type=int, default=2, help="number of rounds to run")
parser.add_argument("--num_morphs", type=int, default=3, help="number of morphs per round")
parser.add_argument("--topk_morphs", type=int, default=2, help="number of top morphs to keep each round")
parser.add_argument("--compute_backend", type=str, default="oop")
args = parser.parse_args()

# Setup and seeding
print(f"Seed: {args.seed}")
random.seed(args.seed)

def random_arxiv_abstract(num_terms: int = 2) -> str:
    query_filepath = os.path.join(PROMPT_DIR, "arxiv_query.txt")
    with open(query_filepath, "r") as f:
        terms = f.read().strip().split(',')
    query = " AND ".join(random.sample(terms, num_terms))
    search_results = arxiv.Search(
        query=query,
        max_results=12,  # Adjust the number if necessary
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    paper = random.choice(list(search_results.results()))
    return f"this paper from arxiv contains a helpful hint:\n{paper.title}\n{paper.summary}"

def load_prompt(prompt_path):
    prompt_filepath = os.path.join(PROMPT_DIR, prompt_path)
    with open(prompt_filepath, "r") as f:
        return f.read()

def morph_nb_prompt(morph: Morph):
    morph_nb_filepath = os.path.join(MORPH_DIR, f"{morph.name}.ipynb")
    with open(morph_nb_filepath, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)

def morphesis(name: str, system: str, prompt: str, output_dir:str = MORPH_DIR) -> Morph:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    reply = completion.choices[0].message.content
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source=reply))
    morph = Morph(0, name)
    print(f"\t🥚\t welcome ~{name}~")
    nb_filepath = os.path.join(output_dir, f"{name}.ipynb")
    with open(nb_filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return morph

def export(morph: Morph) -> Morph:
    export_prompt_filepath = os.path.join(PROMPT_DIR, "export.txt")
    export_prompt = load_prompt(export_prompt_filepath)
    prompt = morph_nb_prompt(morph)
    export_dir = os.path.join(OUTPUT_DIR, morph.name)
    return morphesis(f"export.{morph.name}", export_prompt, prompt, output_dir=export_dir)

def mutate(protomorph: Morph, mutation_prompt_filename: str) -> Morph:
    print("\t 🧫 mutating...")
    neomorph_name = str(uuid.uuid4())[:6]
    neomorph_output_dir = os.path.join(OUTPUT_DIR, neomorph_name)
    os.makedirs(neomorph_output_dir, exist_ok=True)
    mutation_prompt_filepath = os.path.join(MUTATION_PROMPTS_DIR, f"{mutation_prompt_filename}.txt")
    system = load_prompt(mutation_prompt_filepath)
    if random.random() < 0.5:
        print("\t 🍯 adding glazing prompt...")
        glazing_prompt_filepath = os.path.join(PROMPT_DIR, "glazing.txt")
        system += f"\n{load_prompt(glazing_prompt_filepath)}"
    if random.random() < 0.5:
        print("\t 📚 adding arxiv prompt...")
        system += f"\n{random_arxiv_abstract()}"
    if random.random() < 0.5:
        print("\t 📋 adding challenge prompt...")
        challenge_prompt_filepath = os.path.join(PROMPT_DIR, "challenge.txt")
        system += f"\n{load_prompt(challenge_prompt_filepath)}"
    format_prompt_filepath = os.path.join(PROMPT_DIR, "format.txt")
    system += f"\n{load_prompt(format_prompt_filepath)}"
    print(f"\t 👵 parent: ~{protomorph.name}~")
    prompt = morph_nb_prompt(protomorph)
    # write prompt to output dir
    neomorph_prompt_filepath = os.path.join(neomorph_output_dir, "prompt.txt")
    with open(neomorph_prompt_filepath, "w") as f:
        f.write(f"SYSTEM:\n{system}\n\nPROMPT:\n{prompt}")
    neomorph = morphesis(system, prompt, neomorph_name)
    return neomorph

if __name__ == "__main__":

    # Initialize morphs
    morphs: List[Morph] = []
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(MORPH_DIR, f"{protomorph}.ipynb")):
            morphs.append(Morph(0, protomorph))
    print("Morphs:")
    for morph in morphs:
        print(f"\t🧬\t~{morph.name}~")

    # list all files in mutation dir
    mutation_prompts_dir = os.path.join(PROMPT_DIR, "mutations")
    mutation_prompts_filepaths = []
    for mutation_prompt in os.listdir(mutation_prompts_dir):
        mutation_prompts_filepaths.append(os.path.join(mutation_prompts_dir, mutation_prompt))

    # Evolution rounds
    session_id = str(uuid.uuid4())[:6]
    leaderboard_dir = os.path.join(OUTPUT_DIR, f"session.{session_id}")
    os.makedirs(leaderboard_dir, exist_ok=True)
    for round_num in range(args.num_rounds):
        print(f"Round {round_num}")

        # ---- mutation ----
        print("mutating until full morphs...")
        while len(morphs) < args.num_morphs:
            protomorph = random.choice(morphs)
            neomorph = mutate(protomorph, random.choice(MUTATIONS))
            morphs.append(neomorph)

        # ---- selection ----
        print("running morphs...")
        leaderboard = {}
        leaderboard_filepath = os.path.join(leaderboard_dir, f"leaderboard.r{round_num}.yaml")
        for morph in morphs:
            if morph.state == ALREADY_RAN:
                print(f"\t⏩\tSkipping {morph.name} with score {morph.score}")
                continue
            elif morph.state == ERRORED_OUT:
                print(f"\t⏩\tSkipping {morph.name} with errors")
                continue
            else:
                print(f"\t⏯️\tRunning {morph.name}")
            print("Killing stale Docker processes...")
            subprocess.run("docker kill $(docker ps -aq)", shell=True)
            subprocess.run("docker rm $(docker ps -aq)", shell=True)
            time.sleep(2)
            try:
                print("Setting up environment variables...")
                os.environ["MORPH"] = morph.name
                proc = subprocess.Popen(["bash", f"scripts/run.{args.compute_backend}.sh"])
                proc.wait()
                if proc.returncode != 0:
                    print(f"\t❌\tError when running {morph.name}")
                    morph.state = ERRORED_OUT
                    continue
                morph_output_dir = os.path.join(OUTPUT_DIR, morph.name)
                os.makedirs(morph_output_dir, exist_ok=True)
                morph_output_filepath = os.path.join(morph_output_dir, "results.json")
                with open(morph_output_filepath, "r") as f:
                    morph_output = yaml.safe_load(f)
                score = morph_output["accuracy"]
                leaderboard[morph.name] = score
                morph.score = score
                print(f"\t🏁\t{morph.name} scored {score}")
                morph.state = ALREADY_RAN
                export(morph)
            except Exception as e:
                print(f"\t❌\tError when running {morph.name}: {e}")
                continue
        
        # write sorted leaderboard
        leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)}
        with open(leaderboard_filepath, "w") as f:
            yaml.safe_dump(leaderboard, f, default_flow_style=False)

        # ---- elimination ----
        print("Elimination:")
        doomed = []
        for i, morph in enumerate(sorted(morphs, key=lambda m: m.score)):
            score = morph.score
            if i < args.topk_morphs:
                print(f"\t🏆\t{morph.name} is in the top {args.topk_morphs} with score {score}")
            else:
                print(f"\t🗑\t{morph.name} is in the bottom with score {score}")
                doomed.append(morph)

        morphs = [morph for morph in morphs if morph not in doomed]
