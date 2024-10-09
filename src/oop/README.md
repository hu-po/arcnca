
test base container

```bash
docker run --gpus all -it --rm \
gcr.io/kaggle-gpu-images/python \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```

clone repo

```bash
git clone https://github.com/hu-po/cax.git
cd cax
```

test library on raw local

```bash
conda create -n cax python=3.10
conda activate cax
pip install -e '.[dev]'
pytest
```

test base container

```bash
docker run --gpus all -it --rm \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```

run library tests in container

```bash
docker run --gpus all -it --rm \
-v ~/dev/cax:/cax \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c /cax/examples/arc-2024/test.lib.oop.sh
```

run arc 1d test for oop in container

```bash
docker run --gpus all -it --rm \
-v ~/dev/cax:/cax \
-p 8888:8888 \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c /cax/examples/arc-2024/test.1d-arc.oop.sh
```