import argparse
import json
import os

import jax
import jax.numpy as jnp
import optax
from cax.core.ca import CA
from cax.core.perceive.depthwise_conv_perceive import DepthwiseConvPerceive
from cax.core.perceive.kernels import grad_kernel, identity_kernel
from cax.core.update.residual_update import ResidualUpdate
from flax import nnx
from tqdm.auto import tqdm
import wandb


def process_example(example, task_index, ds_size):
    input_data = jnp.squeeze(jnp.array(example["input"], dtype=jnp.int32))
    output_data = jnp.squeeze(jnp.array(example["output"], dtype=jnp.int32))

    assert input_data.shape == output_data.shape

    pad_size = ds_size - input_data.size
    pad_left, pad_right = pad_size // 2, pad_size - pad_size // 2

    input_padded = jnp.pad(input_data, (pad_left, pad_right))
    output_padded = jnp.pad(output_data, (pad_left, pad_right))

    return jnp.expand_dims(
        jnp.concatenate([jnp.array([task_index], dtype=jnp.int32), input_padded, output_padded]), axis=-1
    )


def main():
    parser = argparse.ArgumentParser(description='ARC 2024 Neural Cellular Automata')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--channel_size', type=int, default=32, help='Channel size')
    parser.add_argument('--num_spatial_dims', type=int, default=1, help='Number of spatial dimensions')
    parser.add_argument('--num_kernels', type=int, default=2, help='Number of kernels')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--cell_dropout_rate', type=float, default=0.5, help='Cell dropout rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=128, help='Number of steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--ds_size', type=int, default=128, help='Dataset size')
    parser.add_argument('--num_train_steps', type=int, default=8192, help='Number of training steps')
    parser.add_argument('--print_interval', type=int, default=128, help='Interval for printing logs')
    parser.add_argument('--wandb_project', type=str, default='arc-2024-nca', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default="hug", help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_api_key', type=str, default=None, help='Weights & Biases API key')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the trained model')

    args = parser.parse_args()

    seed = args.seed
    channel_size = args.channel_size
    num_spatial_dims = args.num_spatial_dims
    num_kernels = args.num_kernels
    hidden_size = args.hidden_size
    cell_dropout_rate = args.cell_dropout_rate
    batch_size = args.batch_size
    num_steps = args.num_steps
    learning_rate = args.learning_rate
    ds_size = args.ds_size
    num_train_steps = args.num_train_steps
    print_interval = args.print_interval

    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)

    if args.wandb_api_key:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            'seed': seed,
            'channel_size': channel_size,
            'num_spatial_dims': num_spatial_dims,
            'num_kernels': num_kernels,
            'hidden_size': hidden_size,
            'cell_dropout_rate': cell_dropout_rate,
            'batch_size': batch_size,
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'ds_size': ds_size,
            'num_train_steps': num_train_steps,
            'print_interval': print_interval,
        }
    )

    ds_path = "./1D-ARC/dataset"
    if not os.path.exists(ds_path):
        print('Cloning 1D-ARC dataset...')
        os.system('git clone https://github.com/khalil-research/1D-ARC.git')
    else:
        print('1D-ARC dataset already exists.')

    train_examples = []
    test_examples = []
    task_index_to_name = {}

    for task_index, task_name in enumerate(os.listdir(ds_path)):
        task_index_to_name[task_index] = task_name
        task_path = os.path.join(ds_path, task_name)

        for task_file in os.listdir(task_path):
            with open(os.path.join(task_path, task_file)) as f:
                data = json.load(f)
                for split, examples in [("train", train_examples), ("test", test_examples)]:
                    examples.extend(process_example(ex, task_index, ds_size) for ex in data[split])

    train_tasks = jnp.array(train_examples)
    test_tasks = jnp.array(test_examples)

    task_list = list(task_index_to_name.values())

    def init_state(key):
        sample = jax.random.choice(key, train_tasks)
        task_index, input, target = jnp.split(sample, indices_or_sections=[1, ds_size + 1])
        state = jnp.zeros((ds_size, channel_size))
        state = state.at[..., :1].set(input)
        return state, target, task_index

    def init_state_test(key):
        sample = jax.random.choice(key, test_tasks)
        task_index, input, target = jnp.split(sample, indices_or_sections=[1, ds_size + 1])
        state = jnp.zeros((ds_size, channel_size))
        state = state.at[..., :1].set(input)
        return state, target, task_index

    perceive = DepthwiseConvPerceive(channel_size, rngs, num_kernels=num_kernels, kernel_size=(3,))
    update = ResidualUpdate(
        num_spatial_dims,
        channel_size,
        num_kernels * channel_size + 8,
        (hidden_size,),
        rngs,
        cell_dropout_rate=cell_dropout_rate,
    )
    embed_input = nnx.Embed(num_embeddings=10, features=3, rngs=rngs)
    embed_task = nnx.Embed(num_embeddings=len(task_list), features=8, rngs=rngs)

    class EmbedCA(CA):
        embed_input: nnx.Embed
        embed_task: nnx.Embed

        def __init__(self, perceive, update, embed_input, embed_task):
            super().__init__(perceive, update)
            self.embed_input = embed_input
            self.embed_task = embed_task

    kernel = jnp.concatenate([identity_kernel(ndim=1), grad_kernel(ndim=1)], axis=-1)
    kernel = jnp.expand_dims(jnp.concatenate([kernel] * channel_size, axis=-1), axis=-2)
    perceive.depthwise_conv.kernel = nnx.Param(kernel)
    ca = EmbedCA(perceive, update, embed_input, embed_task)
    params = nnx.state(ca, nnx.Param)
    print("Number of params:", jax.tree_util.tree_reduce(lambda x, y: x + y.size, params, 0))

    lr_sched = optax.linear_schedule(init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=2_000)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_sched),
    )

    update_params = nnx.All(nnx.Param, nnx.PathContains("update"))
    optimizer = nnx.Optimizer(ca, optimizer, wrt=update_params)

    def mse(state, target):
        return jnp.mean(jnp.square(state[..., :3] - target))

    @nnx.jit
    def accuracy_fn(state, target):
        predictions = jnp.argmax(state[..., :3], axis=-1)
        target_labels = jnp.argmax(target, axis=-1)
        correct = jnp.sum(predictions == target_labels)
        total = target_labels.size
        return correct / total

    @nnx.jit
    def loss_fn(ca, state, target, task_index):
        input = state[..., 0]
        input_embed = ca.embed_input(jnp.asarray(input, dtype=jnp.int32))
        task_embed = ca.embed_task(jnp.asarray(task_index, dtype=jnp.int32))
        state = state.at[..., :3].set(input_embed)
        target_embed = ca.embed_input(jnp.asarray(target[..., 0], dtype=jnp.int32))
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
        state, target, task_index = jax.vmap(init_state)(keys)
        loss, grad = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, update_params))(ca, state, target, task_index)
        optimizer.update(grad)
        return loss

    @nnx.jit
    def eval_step(ca, key):
        keys = jax.random.split(key, batch_size)
        state, target, task_index = jax.vmap(init_state_test)(keys)
        accuracy = accuracy_fn(state, target)
        return accuracy

    pbar = tqdm(range(num_train_steps), desc="Training", unit="train_step")
    losses = []
    eval_accuracies = []

    for i in pbar:
        key, subkey = jax.random.split(key)
        loss = train_step(ca, optimizer, subkey)
        losses.append(loss)

        if i % print_interval == 0 or i == num_train_steps - 1:
            avg_loss = sum(losses[-print_interval:]) / len(losses[-print_interval:])
            pbar.set_postfix({"Average Loss": f"{avg_loss:.6f}"})
            wandb.log({'average_loss': avg_loss, 'step': i})

            accuracy = eval_step(ca, subkey)
            eval_accuracies.append(accuracy)
            avg_accuracy = sum(eval_accuracies[-print_interval:]) / len(eval_accuracies[-print_interval:])
            wandb.log({'eval_accuracy': avg_accuracy, 'step': i})

        wandb.log({'loss': loss, 'step': i})

    if args.save_model:
        import pickle
        with open(args.save_model, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model parameters saved to {args.save_model}")

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 8)
    state_init, target, task_index = jax.vmap(init_state_test)(keys)

    input = state_init[..., 0]
    input_embed = ca.embed_input(jnp.asarray(input, dtype=jnp.int32))
    task_embed = ca.embed_task(jnp.asarray(task_index, dtype=jnp.int32))
    state_init = state_init.at[..., :3].set(input_embed)

    state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
    state = nnx.split_rngs(splits=batch_size)(
        nnx.vmap(
            lambda ca, state, task_embed: ca(state, task_embed, num_steps=num_steps, all_steps=True),
            in_axes=(state_axes, 0, 0),
        )
    )(ca, state_init, task_embed)

    state_rgb = jnp.concatenate([jnp.expand_dims(state_init[..., :3], axis=1), state[..., :3]], axis=1)
    task_name = [task_list[int(jnp.squeeze(task_index_i))] for task_index_i in task_index]
    state_rgb_np = jax.device_get(state_rgb)

    wandb.log({
        "examples": [
            wandb.Image(
                state_rgb_np[i],
                caption=f"Task: {task_name[i]}"
            ) for i in range(len(state_rgb_np))
        ]
    })

if __name__ == '__main__':
    main()