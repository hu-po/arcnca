test base container

```bash
docker run --gpus all -it --rm \
gcr.io/kaggle-gpu-images/python \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```

run cax tests inside container

```bash
docker run --gpus all -it --rm \
gcr.io/kaggle-gpu-images/python \
bash
pip install pytest
git clone https://github.com/hu-po/cax.git
pytest
```

run arc 1d test for oop in container (connect to http://localhost:8888 in vscode/browser)

```bash
docker run --gpus all -it --rm \
-p 8888:8888 \
-e WANDB_API_KEY=$WANDB_API_KEY \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "pip install jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
```

this faults on `nnx` in cell 3