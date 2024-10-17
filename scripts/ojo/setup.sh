echo "Setting up ojo environment"
pip install 'pytest>=8.3.2' 'numpy>=1.26.4' 'pillow>=10.4.0' 'msgpack>=1.1.0' 'requests>=2.32.3' 'mediapy>=1.2.2' tqdm
pip install --no-deps 'optax==0.2.3' 'chex==0.1.86' 'flax>=0.9.0' orbax-checkpoint tensorstore 'typing-extensions>=4.2' 'absl-py>=2.1.0' 'toolz>=1.0.0' 'etils[epy]>=1.9.4'
pip install jupyter
pip install wandb