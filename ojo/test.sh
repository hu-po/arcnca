jetson-containers run \
-v /home/$USER/dev/arcnca/$USER:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
-e WANDB_API_KEY=$WANDB_API_KEY \
$(autotag jax) \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute /arcnca/main.ipynb --stdout"