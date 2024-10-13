# 1. Data Augmentation

def augment_data(inputs, outputs):
    augmented_inputs = []
    augmented_outputs = []
    
    for input_grid, output_grid in zip(inputs, outputs):
        # Original data
        augmented_inputs.append(input_grid)
        augmented_outputs.append(output_grid)
        
        # Horizontal flip
        augmented_inputs.append(np.fliplr(input_grid))
        augmented_outputs.append(np.fliplr(output_grid))
        
        # Vertical flip
        augmented_inputs.append(np.flipud(input_grid))
        augmented_outputs.append(np.flipud(output_grid))
        
        # 90-degree rotation
        augmented_inputs.append(np.rot90(input_grid))
        augmented_outputs.append(np.rot90(output_grid))
        
        # 180-degree rotation
        augmented_inputs.append(np.rot90(input_grid, 2))
        augmented_outputs.append(np.rot90(output_grid, 2))
        
        # Add noise (small random perturbations)
        noisy_input = input_grid + np.random.randint(-1, 2, input_grid.shape)
        noisy_input = np.clip(noisy_input, 0, 9)
        augmented_inputs.append(noisy_input)
        augmented_outputs.append(output_grid)
    
    return np.array(augmented_inputs), np.array(augmented_outputs)

# Apply augmentation to training data
augmented_train_inputs, augmented_train_outputs = augment_data(train_inputs, train_outputs)

# 2. Model Architecture

class ImprovedCAX(nnx.Module):
    @nnx.compact
    def __call__(self, x, task_embedding):
        # Expand task embedding to match grid dimensions
        task_embedding = jnp.tile(task_embedding[:, None, None, :], (1, x.shape[1], x.shape[2], 1))
        
        # Concatenate input grid with task embedding
        x = jnp.concatenate([x[..., None], task_embedding], axis=-1)
        
        # Perceive module
        x = nnx.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = jax.nn.relu(x)
        x = nnx.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = jax.nn.relu(x)
        
        # Update module
        x = nnx.Conv(features=64, kernel_size=(1, 1))(x)
        x = jax.nn.relu(x)
        x = nnx.Conv(features=1, kernel_size=(1, 1))(x)
        
        return x.squeeze(-1)

# 3. Training Process

def train_step(state, batch):
    def loss_fn(params):
        inputs, outputs, task_indices = batch
        task_embeddings = state.params['task_embeddings'][task_indices]
        predictions = state.apply_fn({'params': params}, inputs, task_embeddings)
        loss = jnp.mean((predictions - outputs) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 4. Evaluation and Submission

def evaluate(state, inputs, outputs, task_indices):
    task_embeddings = state.params['task_embeddings'][task_indices]
    predictions = state.apply_fn({'params': state.params}, inputs, task_embeddings)
    accuracy = jnp.mean(jnp.all(predictions.round() == outputs, axis=(1, 2)))
    return accuracy

def prepare_submission(state, test_challenges_path):
    with open(test_challenges_path, 'r') as f:
        test_challenges = json.load(f)
    
    submission = {}
    for task_id, task in test_challenges.items():
        task_index = task_id_to_index[task_id]
        task_embedding = state.params['task_embeddings'][task_index]
        
        task_submission = []
        for test_input in task['test']:
            input_grid = jnp.array(pad_grids([np.array(test_input['input'], dtype=np.int32)]))
            predictions = state.apply_fn({'params': state.params}, input_grid, task_embedding[None, ...])
            rounded_predictions = predictions.round().astype(int).tolist()
            
            # Trim padding
            height, width = np.array(test_input['input']).shape
            trimmed_prediction = [row[:width] for row in rounded_predictions[0][:height]]
            
            task_submission.append({
                "attempt_1": trimmed_prediction,
                "attempt_2": trimmed_prediction  # You might want to generate a different second attempt
            })
        
        submission[task_id] = task_submission
    
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

# 5. Additional Improvements

# Implement curriculum learning
def curriculum_learning_schedule(epoch):
    if epoch < 10:
        return 0.5  # Start with easier tasks
    elif epoch < 20:
        return 0.75  # Gradually increase difficulty
    else:
        return 1.0  # Full dataset

# Implement task embedding initialization
def initialize_task_embeddings(num_tasks, embedding_dim=64):
    return jax.random.normal(jax.random.PRNGKey(0), (num_tasks, embedding_dim))

# Main training loop
def train_model():
    model = ImprovedCAX()
    task_embeddings = initialize_task_embeddings(len(task_id_to_index))
    
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.zeros((1, 30, 30)), jnp.zeros((1, 64))),
        tx=optimizer
    )
    state = state.replace(params={**state.params, 'task_embeddings': task_embeddings})
    
    num_epochs = 100
    batch_size = 32
    
    for epoch in range(num_epochs):
        # Apply curriculum learning
        difficulty = curriculum_learning_schedule(epoch)
        num_samples = int(len(augmented_train_inputs) * difficulty)
        
        # Shuffle and batch data
        permutation = jax.random.permutation(jax.random.PRNGKey(epoch), num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch = (
                augmented_train_inputs[batch_indices],
                augmented_train_outputs[batch_indices],
                train_task_indices[batch_indices]
            )
            state, loss = train_step(state, batch)
        
        # Evaluate on validation set
        eval_accuracy = evaluate(state, eval_inputs, eval_outputs, eval_task_indices)
        print(f"Epoch {epoch}, Loss: {loss}, Eval Accuracy: {eval_accuracy}")
    
    return state

# Run training
final_state = train_model()

# Prepare submission
prepare_submission(final_state, test_challenges_path)