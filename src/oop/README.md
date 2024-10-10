tests

```bash
docker run --gpus all -it --rm \
-v /home/oop/dev/arcnca/src/oop:/arcnca \
-e WANDB_API_KEY=$WANDB_API_KEY \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute /arcnca/main.ipynb --stdout"
```