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

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--llm", type=str, default="gpt")
parser.add_argument("--compute", type=str, default="oop") # one of oop, big, ojo
# --- Evolution params
parser.add_argument("--num_players", type=int, default=24)
parser.add_argument("--num_rounds", type=int, default=32)
parser.add_argument("--cull_ratio", type=int, default=4)
args = parser.parse_args()

print("🧫🔬 Starting Evolution")
random.seed(args.seed)
session_id = str(uuid.uuid4())[:6]
base_dir = f"output/{session_id}{datetime.datetime.strftime('
os.makedirs(base_dir, exist_ok=True)
print(f"base directory at {base_dir}")
player_dir = os.path.join(base_dir, "players")
os.makedirs(player_dir, exist_ok=True)
print(f"player directory at {player_dir}")
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
print(f"logs directory at {logs_dir}")
ckpt_dir = os.path.join(base_dir, "ckpt")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"ckpt directory at {ckpt_dir}")

if args.llm == "gpt":
    # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def llm(system: str, prompt: str, temp: float, max_tokens: int):
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

elif args.llm == "codellama":
    # https://replicate.com/meta/codellama-70b-instruct
    import replicate

    def llm(system: str, prompt: str, temp: float, max_tokens: int):
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

# Spin up a Tensorboard instance to monitor training
os.system("pkill -f 'tensorboard'")
tb_proc = subprocess.Popen(["tensorboard", f"--logdir={logs_dir}"])
tb_chrome_proc = subprocess.Popen(["/usr/bin/google-chrome", "http://localhost:6006/"])

# Seed with the players in the local directory "players"
seed_players_dir = os.path.join(os.getcwd(), "players", args.framework)
players = os.listdir(seed_players_dir)
for player in players:
    shutil.copy(os.path.join(seed_players_dir, player), player_dir)
# Remove the player suffix from the player names
players = [x.split(".")[0] for x in players]
# shuffle the players and clip to num_players
random.shuffle(players)
players = players[: args.num_players]
for round in range(args.num_rounds):
    print(f"Starting evolution rounds {round}")
    # reproduce to fill in missing players
    while len(players) < args.num_players:
        run_id = str(uuid.uuid4())[:6]
        print(f"Creating run {run_id}")
        parents = random.sample(players, 2)
        print(f"Reproducing {parents[0]} and {parents[1]}")
        # Add parent names to run_id for easy identification
        run_id = f"{parents[0][:2]}_{parents[1][:2]}_{run_id}"
        # zero-shot
        system_prompt = f"""
You are a expert machine learning research engineer.
You excel at creating new and unique model architectures.
You use {args.framework} and make use of the einops library.
You will be given several example blocks of code.
Create a new block of code inspired by the given blocks.
The block of code should be called `Block` and should be a subclass of `nn.Module`.
Make sure the kwarg `num_classes` is present in the `__init__` method.
Do not explain, return only the working code which will be written directly to a .py file."""
        user_prompt = ""
        for parent in parents:
            parent_filepath = os.path.join(player_dir, f"{parent}.py")
            with open(parent_filepath, "r") as f:
                user_prompt += f"\n{f.read()}"
        reply = llm(system_prompt, user_prompt, 0.9, 512)
        reply = llm(
            """
You are an expert debugging machine.
You fix dim mismatch errors in model architectures.
Return the user provided code with any mistakes removed.
Remove any comments.
Do not explain return only the code.""",
            reply,
            0.7,
            512,
        )
        run_filename = f"{run_id}.py"
        run_filepath = os.path.join(player_dir, run_filename)
        with open(run_filepath, "w") as f:
            # HACK: removes first and last lines
            f.write("\n".join(reply.split("\n")[1:-1]))
        players.append(run_id)

    best_scores = {}
    results_filepath = os.path.join(ckpt_dir, f"results.r{round}.yaml")
    with open(results_filepath, "w") as f:
        yaml.dump({}, f)
    previous_results_filepath = os.path.join(ckpt_dir, f"results.r{round-1}.yaml")
    if os.path.exists(previous_results_filepath):
        with open(previous_results_filepath, "r") as f:
            previous_results = yaml.safe_load(f)
    else:
        previous_results = {}
    for player in players:
        # skip already run players
        if player in previous_results:
            best_scores[player] = previous_results[player]["test_accuracy"]
            continue
        print(f"Running {args.framework} traineval for {player}")
        model_filepath = os.path.join(player_dir, f"{player}.py")
        os.system("docker kill $(docker ps -aq) && docker rm $(docker ps -aq)")
        player_docker_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--gpus=all",
                "-v",
                f"{model_filepath}:/src/model.py",
                "-v",
                f"{ckpt_dir}:/ckpt",
                "-v",
                f"{logs_dir}:/logs",
                "-v",
                f"{data_dir}:/data",
                "-e",
                f"RUN_NAME={player}",
                "-e",
                f"ROUND={round}",
                f"evo_{args.framework}",
            ]
        )
        player_docker_proc.wait()
        if player_docker_proc.returncode != 0:
            print(f"Error occurred when training player {player}")
            best_scores[player] = 0.0
        else:
            print(f"Trained player {player}")
            with open(results_filepath, "r") as f:
                player_results = yaml.safe_load(f)
            best_scores[player] = player_results[player]["test_accuracy"]
        print(f"Player {player} result {best_scores[player]}")

    sorted_players = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted players: {sorted_players}")
    cull_index = len(sorted_players) // args.cull_ratio
    top_players = [x[0] for x in sorted_players[:cull_index]]
    print(f"Top players: {top_players}")
    bot_players = [x[0] for x in sorted_players[-cull_index:]]
    print(f"Bottom players: {bot_players}")
    for player in bot_players:
        os.remove(os.path.join(player_dir, f"{player}.py"))
        print(f"Removed player {player}")
    players = [x for x in players if x not in bot_players]

    # Plot round results
    plot_filepath = os.path.join(ckpt_dir, "test_accuracy_plot.png")
    yaml_files = glob.glob(f"{ckpt_dir}/results.r*.yaml")
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
