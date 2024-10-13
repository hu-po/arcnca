docker run --gpus all -it --rm \
-v /home/$USER/dev/arcnca:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e MORPH_NB_FILEPATH=$MORPH_NB_FILEPATH \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout"