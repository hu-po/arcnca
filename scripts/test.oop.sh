docker run --gpus all -it --rm \
-v /home/$USER/dev/arcnca:/arcnca \
-v /home/$USER/dev/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/$USER/dev/arcnca/output:/kaggle/working \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c "pip install jupyter && jupyter nbconvert --to notebook --execute /arcnca/morphs/test.ipynb --stdout"