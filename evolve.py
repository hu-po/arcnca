"""its morphin' time"""

import argparse
import os
import random
import subprocess
import time
import uuid
import yaml
from dataclasses import dataclass
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--agent", type=str, default="gpt")
parser.add_argument("--tb", type=bool, default=True, help="start tensorboard session")
parser.add_argument("--protomorphs", type=str, help="comma separated list of protomorphs to seed evolution")
parser.add_argument("--num_rounds", type=int, default=32, help="number of rounds to run")
parser.add_argument("--num_morphs", type=int, default=4, help="number of morphs per round")
parser.add_argument("--topk_morphs", type=int, default=2, help="number of top morphs to keep each round")
parser.add_argument("--compute_backend", type=str, default="oop") # one of oop, big, ojo (unless you are hupo this means nothing)
args = parser.parse_args()

print(f"🧫🔬\tseed\t{args.seed}")
random.seed(args.seed)
output_dir = os.path.abspath("output")
morph_dir = os.path.join(output_dir, "morphs")
os.makedirs(morph_dir, exist_ok=True)

if args.agent == "gpt":
    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def agent(system: str, prompt: str, temp: float, max_tokens: int):
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

    def agent(system: str, prompt: str, temp: float, max_tokens: int):
        output = replicate.run(
            "meta/codellama-70b-instruct:a279116fe47a0f65701a8817188601e2fe8f4b9e04a518789655ea7b995851bf",
            input={
                "top_k": 10,
                "top_p": 0.95,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system_prompt": system,
                "repeat_penalty": 1.1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        )
        return output

if args.tb:
    # Spin up a Tensorboard instance to monitor training
    os.system("pkill -f 'tensorboard'")
    tb_proc = subprocess.Popen(["tensorboard", f"--logdir={morph_dir}"])
    tb_chrome_proc = subprocess.Popen(["/usr/bin/google-chrome", "http://localhost:6006/"])

@dataclass(order=True)
class Morph:
    score: float
    name: str

morphs: List[Morph] = []
if not args.protomorphs:
    morphs.append(Morph(-1, "diffnca")) # -1 means it has yet to be scored
else:
    morphs = []
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(morph_dir, protomorph)):
            morphs.append(Morph(-1, protomorph))

def make_morph_notebook(morph: Morph) -> str:
    morph_filepath = os.path.join(morph_dir, protomorph, "code.py")
    with open(morph_filepath, "r") as f:
        raw_code = f.read()
    with open("notebooks/base.ipynb", "r") as f:
        raw_base_notebook = f.read()
    morph_nb_filepath = os.path.join(morph_dir, protomorph, "notebook.ipynb")
    # replaces cell containing #<cell> inside base notebook
    # with the code in the morph's code.py file
    with open(morph_nb_filepath, "w") as f:
        f.write(raw_base_notebook.replace("#<cell>", raw_code))
    return morph_nb_filepath

print("morphs:")
for morphs in morphs:
    print(f"\t🧬\t{protomorph}")

for round in range(args.num_rounds):
    print(f"round {round}")

    # ---- mutation

    print("mutation:")
    while len(morphs) < args.num_morphs:    
        protomorph = random.choice(morphs, weights=[morph.score for morph in morphs])
        protomorph_filepath = os.path.join(morph_dir, protomorph, "code.py")
        protomorph_as_text = "<cell>"
        with open(protomorph_filepath, "r") as f:
            protomorph_as_text += f"{f.read()}"
        protomorph_as_text += "</cell>"

        neomorph = str(uuid.uuid4())[:6]
        neomorph_dir = os.path.join(morph_dir, neomorph)
        neomorph_filepath = os.path.join(neomorph_dir, "code.py")
        os.makedirs(neomorph_dir, exist_ok=True)
        print(f"\t🧬\t{protomorph} has spawned {neomorph}")
        
        with open('prompt.txt', "r") as f:
            prompt = f.read()
        with open('notebooks/base.ipynb', "r") as f:
            base = f.read()
        prompt += f"<notebook>\n{base}\n</notebook>"
        reply = agent(prompt, protomorph_as_text, 0.7, 512)
        with open(neomorph_filepath, "w") as f:
            import pdb; pdb.set_trace()
            # HACK: removes first and last lines
            f.write("\n".join(reply.split("\n")[1:-1]))
        morphs.append(Morph(-1, neomorph))

    # ---- selection

    print("selection:")
    leaderboard = {}
    leaderboard_filepath = os.path.join(output_dir, f"leaderboard.r{round}.yaml")
    for morph in morphs:
        if morph.score != -1:
            print(f"\t⏩\t skipping {morph.name} with score {morph.score}")
            continue
        elif morph.score != -2:
            print(f"\t⏩\t skipping {morph.name} with errors")
            continue
        print(f"\t⏯️\t running {morph.name}")
        morph_filepath = os.path.join(morph_dir, morph, "code.py")
        print("killing stale docker processes ... ")
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        time.sleep(2)
        print("setting up environment variables ...")
        os.environ["MORPH"] = morph
        os.environ["MORPH_NB_FILEPATH"] = make_morph_notebook(morph)
        print(f"\tMORPH_NB_FILEPATH\t{os.environ['MORPH_NB_FILEPATH']}")
        print("running docker ...")
        proc = subprocess.Popen(["bash", f"scripts/run.{args.compute_backend}.sh"])
        proc.wait()
        if proc.returncode != 0:
            print(f"\t❌\terror when running {morph}")
            score = -2
            continue
        try:
            morph_output_filepath = os.path.join(morph_dir, morph, "output.yaml")
            with open(morph_output_filepath, "r") as f:
                morph_output = yaml.safe_load(f)
            score = morph_output["test_accuracy"]
        except Exception as e:
            print(f"\t❌\terror when running {morph}: {e}")
            score = -2
        leaderboard[morph] = score
        morph.score = score
        print(f"\t🏁\t{morph.name} scored {score}")
    with open(leaderboard_filepath, "w") as f:
        yaml.dump(leaderboard, f)

    doomed = []
    for i, (morph, score) in enumerate(sorted(morphs)):
        if i < args.topk_morphs:
            print(f"\t🏆\t{morph.name} is in the top {args.topk_morphs} with score {score}")
        else:
            print(f"\t🗑\t{morph.name} is in the bottom with score {score}")
            doomed.append(morph)
    morphs = [morph for morph in morphs if morph not in doomed]