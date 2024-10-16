import argparse

from evolve import export, mutate, Morph

parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="protomorph to mutate", default="base")
args = parser.parse_args()

neomorph = mutate(Morph(0, args.morph))
export(neomorph)