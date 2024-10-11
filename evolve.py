"""

lifespan
multiple docker process spawning and running, potentially dying but that is okay. monitoring gpu utilization to determine whether more can be spawned.

"""

import argparse
import base64
import glob
import os
import requests
import random
import shutil
import subprocess
import time
import uuid
import yaml
import datetime
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

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

from dataclasses import dataclass
import bisect

@dataclass(order=True)
class Morph:
    score: float
    name: str  # Name is secondary in sorting

morphs: List[Morph] = []
if not args.protomorphs:
    morphs.append(Morph(-1, "base")) # -1 means it has yet to be scored
else:
    morphs = []
    for protomorph in args.protomorphs.split(","):
        if os.path.exists(os.path.join(morph_dir, protomorph)):
            morphs.append(Morph(-1, protomorph))

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
        neomorph = str(uuid.uuid4())[:6]
        neomorph_dir = os.path.join(morph_dir, neomorph)
        neomorph_filepath = os.path.join(neomorph_dir, "code.py")
        os.makedirs(neomorph_dir, exist_ok=True)
        print(f"\t🧬\t{protomorph} has spawned {neomorph}")
        protomorph_as_text = ""
        with open(protomorph_filepath, "r") as f:
            protomorph_as_text += f"\n{f.read()}"
        # zero-shot
        reply = agent("""
You are a expert machine learning research engineer.
You excel at creating new and unique model architectures.
You will be given several example blocks of code.
Create a new block of code inspired by the given blocks.
The block of code should be called `Block` and should be a subclass of `nn.Module`.
Make sure the kwarg `num_classes` is present in the `__init__` method.
Do not explain, return only the working code which will be written directly to a .py file.""",
            protomorph_as_text, 0.7, 512)
        reply = agent("""
You are an expert debugging machine.
You fix dim mismatch errors in model architectures.
Return the user provided code with any mistakes removed.
Remove any comments.
Do not explain return only the code.""",
            reply, 0.7, 512)
        with open(neomorph_filepath, "w") as f:
            # HACK: removes first and last lines
            f.write("\n".join(reply.split("\n")[1:-1]))
        morphs.append(Morph(-1, neomorph))

    # ---- selection

    print("selection:")
    leaderboard = {}
    leaderboard_filepath = os.path.join(output_dir, f"leaderboard.r{round}.yaml")
    with open(leaderboard_filepath, "w") as f:
        yaml.dump({m.name : m.score for m in morphs}, f)
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
        os.environ["MORPH"] = morph
        print("running docker ...")
        proc = subprocess.Popen(["bash", f"scripts/{args.compute_backend}.sh"])
        proc.wait()
        if proc.returncode != 0:
            print(f"\t❌\terror occurred when training morph {morph}")
            leaderboard[morph] = -2
        else:
            print("looking for results")
            try:
                morph_output_filepath = os.path.join(morph_dir, morph, "output.yaml")
                with open(morph_output_filepath, "r") as f:
                    morph_output = yaml.safe_load(f)
                morph_score = morph_output["test_accuracy"]
            except
            leaderboard[morph] = morph_leaderboard[morph]["test_accuracy"]
        print(f"Player {morph} result {leaderboard[morph]}")

    sorted_morphs = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted morphs: {sorted_morphs}")
    cull_index = len(sorted_morphs) // args.cull_ratio
    top_morphs = [x[0] for x in sorted_morphs[:cull_index]]
    print(f"Top morphs: {top_morphs}")
    bot_morphs = [x[0] for x in sorted_morphs[-cull_index:]]
    print(f"Bottom morphs: {bot_morphs}")
    for morph in bot_morphs:
        os.remove(os.path.join(morph_dir, f"{morph}.py"))
        print(f"Removed morph {morph}")
    morphs = [x for x in morphs if x not in bot_morphs]

    # Plot round leaderboard
    plot_filepath = os.path.join(ckpt_dir, "test_accuracy_plot.png")
    yaml_files = glob.glob(f"{ckpt_dir}/leaderboard.r*.yaml")
    rounds = []
    test_acc = []
    for file in yaml_files:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        round_number = int(file.split(".")[-2].split("r")[-1])
        for key in data:
            rounds.append(round_number)
            test_acc.append(data[key]["test_accuracy"])

    plt.scatter(rounds, test_acc)
    plt.xlabel("round")
    plt.ylabel("acc")
    plt.title("evolution")
    plt.xlim(0, 32)
    plt.ylim(0, 1)
    plt.savefig(plot_filepath)
