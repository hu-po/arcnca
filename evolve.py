import argparse
import os
import random
import subprocess
import time
import uuid
import yaml
from typing import List

from utils import Morph, ALREADY_RAN, ERRORED_OUT, MORPH_DIR, OUTPUT_DIR, PROMPT_DIR, apply_prompt_to_morph, load_prompt

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="gpt")
parser.add_argument("--tb", action="store_true", help="start tensorboard session")
parser.add_argument("--protomorphs", type=str, default="test", help="comma separated list of protomorphs to seed evolution")
parser.add_argument("--num_rounds", type=int, default=2, help="number of rounds to run")
parser.add_argument("--num_morphs", type=int, default=3, help="number of morphs per round")
parser.add_argument("--topk_morphs", type=int, default=2, help="number of top morphs to keep each round")
parser.add_argument("--compute_backend", type=str, default="oop")
args = parser.parse_args()

# Setup and seeding
print(f"Seed: {args.seed}")
random.seed(args.seed)

# Initialize morphs
morphs: List[Morph] = []
for protomorph in args.protomorphs.split(","):
    if os.path.exists(os.path.join(MORPH_DIR, f"{protomorph}.ipynb")):
        morphs.append(Morph(0, protomorph))
print("Morphs:")
for morph in morphs:
    print(f"\t🧬\t{morph.name}")

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
    print("Mutation:")
    while len(morphs) < args.num_morphs:
        protomorph = random.choice(morphs) # TODO: weighted choice based on score
        neomorph_name = str(uuid.uuid4())[:6]
        print(f"\t🧬\t{protomorph.name} has spawned {neomorph_name}")
        prompt = ""
        if random.random() < 0.5:
            print("Adding glazing prompt...")
            glazing_prompt_filepath = os.path.join(PROMPT_DIR, "glazing.txt")
            prompt += load_prompt(glazing_prompt_filepath)
        mutation_prompt_filepath = random.choice(mutation_prompts_filepaths)
        prompt += load_prompt(mutation_prompt_filepath)
        if random.random() < 0.5:
            print("Adding challenge prompt...")
            challenge_prompt_filepath = os.path.join(PROMPT_DIR, "challenge.txt")
            prompt += load_prompt(challenge_prompt_filepath)
        # write prompt to output dir
        neomorph_output_dir = os.path.join(OUTPUT_DIR, neomorph_name)
        os.makedirs(neomorph_output_dir, exist_ok=True)
        neomorph_prompt_filepath = os.path.join(neomorph_output_dir, "prompt.txt")
        with open(neomorph_prompt_filepath, "w") as f:
            f.write(prompt)
        neomorph = apply_prompt_to_morph(protomorph, prompt, neomorph_name)
        morphs.append(neomorph)

    # ---- selection ----
    print("Selection:")
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
            morph_output_filepath = os.path.join(morph_output_dir, "results.json")
            with open(morph_output_filepath, "r") as f:
                morph_output = yaml.safe_load(f)
            score = morph_output["accuracy"]
            leaderboard[morph.name] = score
            morph.score = score
            print(f"\t🏁\t{morph.name} scored {score}")
            export_prompt_filepath = os.path.join(PROMPT_DIR, "export.txt")
            prompt = load_prompt(export_prompt_filepath)
            apply_prompt_to_morph(morph, prompt, f"export.{morph.name}")
            morph.state = ALREADY_RAN
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
