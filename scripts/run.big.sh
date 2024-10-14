sudo docker run --gpus all -it --rm \
-v /home/ubuntu/arcnca:/arcnca \
-v /home/ubuntu/arcnca/data:/kaggle/input/arc-prize-2024 \
-v /home/ubuntu/arcnca/output:/kaggle/working \
-e COMPUTE_BACKEND="big" \
-e MORPH=$MORPH \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e WANDB_ENTITY="hug" \
-e WANDB_PROJECT="arcnca" \
nvcr.io/nvidia/jax:24.04-py3 \
bash -c 'pip install jupyter && jupyter nbconvert --to notebook --execute $MORPH_NB_FILEPATH --stdout'