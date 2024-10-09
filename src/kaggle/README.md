
test base container

```bash
docker run --gpus all -it --rm \
gcr.io/kaggle-gpu-images/python \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```