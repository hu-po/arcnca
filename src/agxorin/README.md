- https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax

test base container

```bash
jetson-containers run \
$(autotag jax) \
bash -c "python3 -c 'import jax; print(jax.devices())'"
```

run library tests

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
$(autotag jax) \
bash -c /cax/examples/arc-2024/test.lib.ojo.sh
```

run 1d-arc notebook from inside container

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-p 8888:8888 \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/test.1d-arc.ojo.sh"
```

run arc-2024 notebook from inside container

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-p 8888:8888 \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/train.arc-2024.nb.ojo.sh"
```

run arc-2024 script from inside container

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/train.arc-2024.ojo.sh"
```

run a sweep

```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax)
```
```bash
pip install wandb
wandb login
wandb sweep -e hug -p arc-2024-nca /cax/examples/arc-2024/sweep1.yaml
```
```bash
jetson-containers run \
-v ~/dev/cax:/cax \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "/cax/examples/arc-2024/train.arc-2024.sweep.ojo.sh"
```