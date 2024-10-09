
test base container

```bash
docker run --gpus all -it --rm \
gcr.io/kaggle-gpu-images/python \
bash -c "nvidia-smi && python3 -c 'import jax; print(jax.devices())'"
```

https://github.com/maxencefaldor/cax/blob/575376d2b7784ecbd50e4aad68121950ad568fba/examples/1d_arc_nca.ipynb
