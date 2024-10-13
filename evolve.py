import argparse
import os
import random
import subprocess
import time
import uuid
import yaml
import datetime
import logging
from dataclasses import dataclass
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="gpt")
parser.add_argument("--tb", action="store_true", help="start tensorboard session")
parser.add_argument("--protomorphs", type=str, help="comma separated list of protomorphs to seed evolution")
parser.add_argument("--num_rounds", type=int, default=32, help="number of rounds to run")
parser.add_argument("--num_morphs", type=int, default=4, help="number of morphs per round")
parser.add_argument("--topk_morphs", type=int, default=2, help="number of top morphs to keep each round")
parser.add_argument("--compute_backend", type=str, default="oop")
args = parser.parse_args()

# Setup and seeding
logging.info(f"Seed: {args.seed}")
random.seed(args.seed)
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

# Initialize morphs
morphs: List[Morph] = []
if not args.protomorphs:
    morphs.append(Morph(0, "diffnca"))
else:
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(morph_dir, protomorph)):
            morphs.append(Morph(0, protomorph))
logging.info("Morphs:")
for morph in morphs:
    logging.info(f"\t🧬\t{morph.name}")

def make_morph_notebook(morph: Morph) -> str:
    morph_filepath = os.path.join(morph_dir, morph.name, "code.py")
    with open(morph_filepath, "r") as f:
        raw_code = f.read()
    with open("morphs/base.ipynb", "r") as f:
        raw_base_notebook = f.read()
    morph_nb_filepath = os.path.join(morph_dir, morph.name, "notebook.ipynb")
    # Replace cell containing #<cell> inside base notebook
    with open(morph_nb_filepath, "w") as f:
        f.write(raw_base_notebook.replace("#<cell>", raw_code))
    return morph_nb_filepath

# Agent function based on the chosen agent type
TEMPERATURE = 0.7
MAX_TOKENS = 512
REPEAT_PENALTY = 1.1
def agent(system: str, prompt: str, temp: float, max_tokens: int):
    if args.agent == "gpt":
        # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4-1106-preview",
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    elif args.agent == "codellama":
        # https://replicate.com/meta/codellama-70b-instruct
        import replicate
        output = replicate.run(
            "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf",
            input={
                "top_k": 10,
                "top_p": 0.95,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system_prompt": system,
                "repeat_penalty": REPEAT_PENALTY,
            },
        )
        return output

# Evolution rounds
for round_num in range(args.num_rounds):
    logging.info(f"Round {round_num}")

    # ---- mutation ----
    logging.info("Mutation:")
    while len(morphs) < args.num_morphs:
        protomorph = random.choices(morphs, weights=[morph.score for morph in morphs])[0]
        protomorph_filepath = os.path.join(morph_dir, protomorph.name, "code.py")
        protomorph_as_text = "<cell>"
        with open(protomorph_filepath, "r") as f:
            protomorph_as_text += f"{f.read()}"
        protomorph_as_text += "</cell>"

        neomorph = str(uuid.uuid4())[:6]
        neomorph_dir = os.path.join(morph_dir, neomorph)
        neomorph_filepath = os.path.join(neomorph_dir, "code.py")
        os.makedirs(neomorph_dir, exist_ok=True)
        logging.info(f"\t🧬\t{protomorph.name} has spawned {neomorph}")

        if not os.path.exists('prompt.txt') or not os.path.exists('morphs/base.ipynb'):
            raise FileNotFoundError("Required files (prompt.txt or base.ipynb) are missing")

        with open('prompt.txt', "r") as f:
            prompt = f.read()
        with open('morphs/base.ipynb', "r") as f:
            base = f.read()
        prompt += f"<notebook>\n{base}\n</notebook>"
        reply = agent(prompt, protomorph_as_text, TEMPERATURE, MAX_TOKENS)
        with open(neomorph_filepath, "w") as f:
            # HACK: removes first and last lines
            f.write("\n".join(reply.split("\n")[1:-1]))
        morphs.append(Morph(0, neomorph))

    # ---- selection ----
    logging.info("Selection:")
    leaderboard = {}
    leaderboard_filepath = os.path.join(output_dir, f"leaderboard.r{round_num}.yaml")
    for morph in morphs:
        if morph.state == ALREADY_RAN:
            logging.info(f"\t⏩\tSkipping {morph.name} with score {morph.score}")
            continue
        elif morph.state == ERRORED_OUT:
            logging.info(f"\t⏩\tSkipping {morph.name} with errors")
            continue
        else:
            logging.info(f"\t⏯️\tRunning {morph.name}")
        logging.info("Killing stale Docker processes...")
        subprocess.run(["docker", "kill", "$(docker ps -aq)"], shell=True)
        subprocess.run(["docker", "rm", "$(docker ps -aq)"], shell=True)
        time.sleep(2)
        logging.info("Setting up environment variables...")
        os.environ["MORPH"] = morph.name
        morph_nb_filepath = make_morph_notebook(morph)
        os.environ["MORPH_NB_FILEPATH"] = morph_nb_filepath
        logging.info(f"Environment variable MORPH_NB_FILEPATH: {os.environ['MORPH_NB_FILEPATH']}")
        morph_output_dir = os.path.join(morph_dir, morph.name, datetime.datetime.now().isoformat())
        os.makedirs(morph_output_dir, exist_ok=True)
        morph_output_dir = os.path.join(morph_output_dir, "output.yaml")
        os.environ["MORPH_OUTPUT_DIR"] = morph_output_dir
        logging.info(f"Environment variable MORPH_OUTPUT_DIR: {os.environ['MORPH_OUTPUT_DIR']}")
        try:
            proc = subprocess.Popen(["bash", f"scripts/run.{args.compute_backend}.sh"])
            proc.wait()
            if proc.returncode != 0:
                logging.error(f"\t❌\tError when running {morph.name}")
                morph.state = ERRORED_OUT
                continue
            with open(morph_output_dir, "r") as f:
                morph_output = yaml.safe_load(f)
            score = morph_output["test_acc"]
            leaderboard[morph.name] = score
            morph.score = score
            logging.info(f"\t🏁\t{morph.name} scored {score}")
        except Exception as e:
            logging.error(f"\t❌\tError when running {morph.name}: {e}")
            continue
    
    # write sorted leaderboard
    leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)}
    with open(leaderboard_filepath, "w") as f:
        yaml.safe_dump(leaderboard, f, default_flow_style=False)

    # ---- elimination ----
    logging.info("Elimination:")
    doomed = []
    for i, morph in enumerate(sorted(morphs, key=lambda m: m.score)):
        score = morph.score
        if i < args.topk_morphs:
            logging.info(f"\t🏆\t{morph.name} is in the top {args.topk_morphs} with score {score}")
        else:
            logging.info(f"\t🗑\t{morph.name} is in the bottom with score {score}")
            doomed.append(morph)

    morphs = [morph for morph in morphs if morph not in doomed]
