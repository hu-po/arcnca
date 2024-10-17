jetson-containers run \
-v /home/ojo/dev/arcnca/:/arcnca \
-v /home/ojo/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/ojo/dev/arcnca/output:/kaggle/working \
-e COMPUTE_BACKEND="ojo" \
-e MORPH=$1 \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "./arcnca/scripts/.\$COMPUTE_BACKEND/setup.sh && jupyter nbconvert --to notebook --execute /arcnca/morphs/\$MORPH.ipynb --stdout"