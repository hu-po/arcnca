# arcnca 🧫🔬

ARC-AGI with NCA

Abstract Reasoning Challenge with Neural Cellular Automata

- https://arcprize.org/
- https://www.kaggle.com/competitions/arc-prize-2024
- https://github.com/maxencefaldor/cax/blob/main/examples/1d_arc_nca.ipynb

## Overview

various types of compute were used

- `ojo` - a local nvidia agx orin
- `oop` - a local linux pc with nvidia 3090
- `big` - a cloud instance with 1xH100
- `kag` - a cloud kaggle nb with 4xT4

## Dependencies and Data

use the minimal `notebooks/test.ipynb` notebook to run tests on a machine. first ssh into your machine and run the following command:

```bash
git clone https://github.com/hu-po/arcnca
./scripts/test.big.sh
```

the test notebook installs necessary packages, runs `cax` tests, downloads the data, performs dummy jax compute, and saves results to file.

## Submitting to Kaggle

for kaggle you need to "create a notebook" from the ["code" page](https://www.kaggle.com/competitions/arc-prize-2024/code) and paste in the contents of `test.ipynb` then click "save version" and make sure to disable internet. then go to the ["submit" page](https://www.kaggle.com/competitions/arc-prize-2024/submit) and hit "submit prediction".

## Useful Links

- https://arcprize.org/
- https://www.kaggle.com/competitions/arc-prize-2024

- https://distill.pub/2020/growing-ca/
- https://arxiv.org/pdf/2410.02651.pdf
- https://github.com/maxencefaldor/cax
- https://github.com/maxencefaldor/cax/blob/main/examples/1d_arc_nca.ipynb
- https://github.com/maxencefaldor/cax/blob/main/examples/diffusing_nca.ipynb

- https://cloud.lambdalabs.com/instances