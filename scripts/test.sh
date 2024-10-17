#!/bin/bash
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <COMPUTE_BACKEND> <MORPH>"
  exit 1
fi
COMPUTE_BACKEND=$1
MORPH=$2
bash $COMPUTE_BACKEND/run.sh $MORPH