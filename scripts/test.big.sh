sudo docker run --gpus all -it --rm \
-v /home/ubuntu/arcnca:/arcnca \
-v /home/ubuntu/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/ubuntu/arcnca/output:/kaggle/working \
-e MORPH_NB_FILEPATH="/arcnca/morphs/conv.ipynb" \
-e WANDB_ENTITY="hug" \
-e WANDB_PROJECT="arcnca" \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c 'pip install jupyter && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout'