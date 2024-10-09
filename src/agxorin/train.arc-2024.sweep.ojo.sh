pip install --no-deps optax==0.2.3 chex==0.1.86 'flax>=0.9.0'
pip install 'pytest>=8.3.2' 'pillow>=10.4.0' 'msgpack>=1.1.0' orbax-checkpoint tensorstore 'typing-extensions>=4.2' 'absl-py>=2.1.0' 'toolz>=1.0.0' 'etils[epy]>=1.9.4'
pip install -e /cax --no-deps
pip install mediapy tqdm wandb
ln -s $(which python3) /usr/local/bin/python
wandb agent hug/arc-2024-nca/0e4jmudo