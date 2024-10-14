jetson-containers run \
-v /home/$USER/dev/arcnca/:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e MORPH_NB_FILEPATH="/arcnca/morphs/conv.ipynb" \
-e WANDB_ENTITY="hug" \
-e WANDB_PROJECT="arcnca" \
$(autotag jax) \
bash -c 'pip install jupyter && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout'