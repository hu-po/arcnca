import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
# import wandb

import flax
from flax import nnx
import optax
import cax
from cax.core.ca import CA
from cax.core.perceive.depthwise_conv_perceive import DepthwiseConvPerceive
from cax.core.perceive.kernels import grad_kernel, identity_kernel
from cax.core.update.residual_update import ResidualUpdate

def set_training_params(test_mode=True):
    if test_mode:
        batch_size = 2
        num_steps = 2
        num_train_steps = 2
        print_interval = 1
    else:
        batch_size = 16
        num_steps = 12
        num_train_steps = 1000
        print_interval = 50
    return batch_size, num_steps, num_train_steps, print_interval

TEST_MODE = True
batch_size, num_steps, num_train_steps, print_interval = set_training_params(TEST_MODE)

seed = 0
key = jax.random.PRNGKey(seed)
rngs = nnx.Rngs(seed)

channel_size = 32
num_kernels = 2
hidden_size = 256
cell_dropout_rate = 0.5
learning_rate = 1e-3
ds_size = 30

num_tasks = len(task_id_to_index)
print(f"Number of tasks: {num_tasks}")

# Define the NCA model
class EmbedCA(CA):
    embed_input: nnx.Embed
    embed_task: nnx.Embed

    def __init__(self, perceive, update, embed_input, embed_task):
        super().__init__(perceive, update)
        self.embed_input = embed_input
        self.embed_task = embed_task

    def __call__(self, state, task_embed, num_steps=1, all_steps=False):
        steps = []
        for _ in range(num_steps):
            state = self.step(state, task_embed)
            if all_steps:
                steps.append(state)
        if all_steps:
            return jnp.stack(steps)
        else:
            return state


def init_state(inputs, outputs, task_indices, key):
    idx = jax.random.randint(key, (), 0, inputs.shape[0])
    input_grid = inputs[idx]
    target_grid = outputs[idx]
    task_index = task_indices[idx]
    state = jnp.zeros((ds_size, ds_size, channel_size))
    state = state.at[..., 0].set(input_grid)
    return state, target_grid, task_index

def init_nca_model():
    perceive = DepthwiseConvPerceive(channel_size, rngs, num_kernels=num_kernels, kernel_size=(3, 3))
    update = ResidualUpdate(
        num_spatial_dims=2,
        channel_size=channel_size,
        input_size=num_kernels * channel_size + 8,
        hidden_sizes=(hidden_size,),
        rngs=rngs,
        cell_dropout_rate=cell_dropout_rate,
    )
    embed_input = nnx.Embed(num_embeddings=10, features=3, rngs=rngs)
    embed_task = nnx.Embed(num_embeddings=num_tasks, features=8, rngs=rngs)
    ca = EmbedCA(perceive, update, embed_input, embed_task)
    return ca

ca = init_nca_model()

identity = identity_kernel(ndim=2)
gradient = grad_kernel(ndim=2)
base_kernel = jnp.concatenate([identity, gradient], axis=-1)
base_kernel = base_kernel[:, :, None, :]
features = channel_size * num_kernels
tiles = int(np.ceil(features / base_kernel.shape[-1]))
kernel = jnp.tile(base_kernel, (1, 1, 1, tiles))
kernel = kernel[:, :, :, :features]
ca.perceive.depthwise_conv.kernel = nnx.Param(kernel)
params = nnx.state(ca, nnx.Param)

def init_optimizer(ca):
    lr_sched = optax.linear_schedule(init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=2000)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_sched),
    )
    update_params = nnx.All(nnx.Param, nnx.PathContains("update"))
    optimizer = nnx.Optimizer(ca, optimizer, wrt=update_params)
    return optimizer, update_params

optimizer, update_params = init_optimizer(ca)

def mse(state, target):
    return jnp.mean(jnp.square(state[..., :3] - target))

@nnx.jit
def accuracy_fn(state, target):
    predictions = jnp.argmax(state[..., :3], axis=-1)
    correct = jnp.sum(predictions == target)
    total = target.size
    return correct / total

@nnx.jit
def loss_fn(ca, state, target, task_index):
    input_grid = state[..., 0]
    input_embed = ca.embed_input(jnp.asarray(input_grid, dtype=jnp.int32))
    task_embed = ca.embed_task(jnp.asarray(task_index, dtype=jnp.int32))
    state = state.at[..., :3].set(input_embed)
    target_embed = ca.embed_input(jnp.asarray(target, dtype=jnp.int32))
    state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
    state = nnx.split_rngs(splits=batch_size)(
        nnx.vmap(
            lambda ca, state, task_embed: ca(state, task_embed, num_steps=num_steps),
            in_axes=(state_axes, 0, 0),
        )
    )(ca, state, task_embed)
    loss = mse(state, target_embed)
    return loss

@nnx.jit
def train_step(ca, optimizer, key):
    keys = jax.random.split(key, batch_size)
    state, target, task_index = jax.vmap(lambda k: init_state(train_inputs, train_outputs, train_task_indices, k))(keys)
    loss, grad = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, update_params))(ca, state, target, task_index)
    optimizer.update(grad)
    return loss

@nnx.jit
def eval_step(ca, key):
    keys = jax.random.split(key, batch_size)
    state, target, task_index = jax.vmap(lambda k: init_state(eval_inputs, eval_outputs, eval_task_indices, k))(keys)
    accuracy = accuracy_fn(state, target)
    return accuracy

pbar = tqdm(range(num_train_steps), desc="Training", unit="step")
losses = []
eval_accuracies = []

for i in pbar:
    key, subkey = jax.random.split(key)
    loss = train_step(ca, optimizer, subkey)
    losses.append(loss)

    if i % print_interval == 0 or i == num_train_steps - 1:
        avg_loss = sum(losses[-print_interval:]) / len(losses[-print_interval:])
        pbar.set_postfix({"Avg Loss": f"{avg_loss:.6f}"})
        accuracy = eval_step(ca, subkey)
        eval_accuracies.append(accuracy)
        avg_accuracy = sum(eval_accuracies[-print_interval:]) / len(eval_accuracies[-print_interval:])
        print(f"Step {i}, Avg Loss: {avg_loss:.6f}, Eval Acc: {avg_accuracy:.4f}")

def prepare_submission(ca, test_challenges_path):
    with open(test_challenges_path, 'r') as f:
        test_challenges = json.load(f)

    submission = {}
    for task_id, task in test_challenges.items():
        test_pairs = task['test']
        outputs = []
        for test_input in test_pairs:
            input_grid = np.array(test_input['input'], dtype=np.int32)
            padded_input = pad_grids([input_grid])[0]
            state = np.zeros((ds_size, ds_size, channel_size), dtype=np.float32)
            state[..., 0] = padded_input
            input_embed = ca.embed_input(jnp.asarray(state[..., 0], dtype=jnp.int32))
            task_index = task_id_to_index.get(task_id, 0)
            task_embed = ca.embed_task(jnp.asarray(task_index, dtype=jnp.int32))
            state = jnp.array(state)
            state = state.at[..., :3].set(input_embed)
            state1 = ca(state, task_embed, num_steps=num_steps)
            output_grid1 = jnp.argmax(state1[..., :3], axis=-1).astype(int)
            output_grid1 = output_grid1[:input_grid.shape[0], :input_grid.shape[1]]
            state2 = ca(state, task_embed, num_steps=num_steps + 64)
            output_grid2 = jnp.argmax(state2[..., :3], axis=-1).astype(int)
            output_grid2 = output_grid2[:input_grid.shape[0], :input_grid.shape[1]]
            outputs.append({
                "attempt_1": output_grid1.tolist(),
                "attempt_2": output_grid2.tolist()
            })
        submission[task_id] = outputs

    with open('submission.json', 'w') as f:
        json.dump(submission, f)

test_challenges_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'
prepare_submission(ca, test_challenges_path)