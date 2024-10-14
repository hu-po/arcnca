jetson-containers run \
-v /home/$USER/dev/arcnca/:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e COMPUTE_BACKEND="ojo" \
-e MORPH=$MORPH \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e WANDB_ENTITY="hug" \
-e WANDB_PROJECT="arcnca" \
$(autotag jax) \
bash -c 'pip install jupyter flax && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout'