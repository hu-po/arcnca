jetson-containers run \
-v /home/$USER/dev/arcnca/:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e MORPH_NB_FILEPATH=$MORPH_NB_FILEPATH \
-e MORPH_OUTPUT_DIR=$MORPH_OUTPUT_DIR \
$(autotag jax) \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout"