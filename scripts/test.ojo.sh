jetson-containers run \
-v /home/$USER/dev/arcnca/:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e COMPUTE_BACKEND="ojo" \
-e MORPH="conv" \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e WANDB_ENTITY="hug" \
-e WANDB_PROJECT="arcnca" \
$(autotag jax) \
bash -c "./arcnca/scripts/setup.\$COMPUTE_BACKEND.sh && jupyter nbconvert --to notebook --execute /arcnca/morphs/\$MORPH.ipynb --stdout"