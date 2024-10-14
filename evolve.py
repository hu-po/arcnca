import argparse
import os
import random
import subprocess
import time
import uuid
import yaml
from typing import List

from utils import Morph, ALREADY_RAN, ERRORED_OUT, MORPH_DIR, OUTPUT_DIR, PROMPT_DIR, apply_prompt_to_morph

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="gpt")
parser.add_argument("--tb", action="store_true", help="start tensorboard session")
parser.add_argument("--protomorphs", type=str, help="comma separated list of protomorphs to seed evolution")
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
if not args.protomorphs:
    morphs.append(Morph(0, "conv"))
else:
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(MORPH_DIR, protomorph)):
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
for round_num in range(args.num_rounds):
    print(f"Round {round_num}")

    # ---- mutation ----
    print("Mutation:")
    while len(morphs) < args.num_morphs:
        protomorph = random.choice(morphs) # TODO: weighted choice based on score
        neomorph_name = str(uuid.uuid4())[:6]
        print(f"\t🧬\t{protomorph.name} has spawned {neomorph_name}")
        neomorph = apply_prompt_to_morph(protomorph, random.choice(mutation_prompts_filepaths), neomorph_name)
        morphs.append(neomorph)

    # ---- selection ----
    print("Selection:")
    leaderboard = {}
    leaderboard_filepath = os.path.join(OUTPUT_DIR, f"leaderboard.r{round_num}.yaml")
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
        subprocess.run(["docker", "kill", "$(docker ps -aq)"], shell=True)
        subprocess.run(["docker", "rm", "$(docker ps -aq)"], shell=True)
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
            morph_output_filepath = os.path.join(OUTPUT_DIR, f"{morph.name}.yaml")
            with open(morph_output_filepath, "r") as f:
                morph_output = yaml.safe_load(f)
            score = morph_output["test_acc"]
            leaderboard[morph.name] = score
            morph.score = score
            print(f"\t🏁\t{morph.name} scored {score}")
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
