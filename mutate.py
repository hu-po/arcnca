import argparse

from evolve import export, mutate, Morph

parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="protomorph to mutate", default="base")
parser.add_argument("--mutation", help="filename of prompt to mutate with", default="open_ended")
args = parser.parse_args()

neomorph = mutate(Morph(0, args.morph), args.mutation)
export(neomorph)