tests

```bash
jetson-containers run \
-v /home/ojo/dev/arcnca/src/ojo:/arcnca \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute /arcnca/main.ipynb --stdout | tee /arcnca/logs/output.log"
```