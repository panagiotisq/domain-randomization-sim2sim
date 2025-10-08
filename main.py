import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.distributions import Normal

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pygame

from env import Point2DEnv
from model import ActorCritic


# Function to compute Generalized Advantage Estimation (GAE) and returns
def compute_returns(rewards, dones, values, gamma=0.99, lam=0.95):
    returns = []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
        next_value = values[step]
    return torch.tensor(returns)



# PPO training parameters
epochs = 100
lr = 1e-4
steps_per_epoch = 1024
mini_batch_size = 64
clip_epsilon = 0.1
gamma = 0.99
lam = 0.95
reward_per_epoch = []
best_avg_reward = -float("inf")
best_model_state = None
successes_per_epoch = []
best_model_state = None
patience = epochs/4
epochs_without_improvement = 0
initial_std = 0.5
final_std = 0.1


# Environment and model setup
env = Point2DEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy = ActorCritic(obs_dim=4, act_dim=2)

# if os.path.exists("ppo_best.pt"):
#     policy.load_state_dict(torch.load("ppo_best.pt"))
#     policy.eval()
#     print("✅ Pre-trained policy loaded.")

# Optimizer and learning rate scheduler
optimizer = optim.Adam(policy.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)


#PPO Training Loop
for epoch in range(epochs):
    success_count = 0
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    obs, _ = env.reset()
    for step in range(steps_per_epoch):
        # Get action distribution and value estimate
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        logits, value = policy(obs_tensor)

        # Dynamic standard deviation schedule for exploration
        std = initial_std - (initial_std - final_std) * (epoch / epochs)
        dist = Normal(logits, torch.tensor(std))
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Environment step
        action_np = action.numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated = env.step(action_np)
        done = terminated

        # Store transition
        obs_buf.append(obs)
        act_buf.append(action_np)
        logp_buf.append(log_prob.detach().numpy())
        val_buf.append(value.item())
        rew_buf.append(reward)
        done_buf.append(done)

        obs = next_obs

        if epoch > 95*epochs/100:
            # Update plot and process GUI events so window refreshes during training:
            if step % 5 == 0 or done:
                env.render_3d()

        if done:
            # Check for success
            if np.linalg.norm(env.position - env.goal) <env.threshold:
                success_count += 1
            obs, _ = env.reset()

    # Convert lists to tensors
    obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(act_buf), dtype=torch.float32)
    logp_tensor = torch.tensor(np.array(logp_buf), dtype=torch.float32)
    val_tensor = torch.tensor(np.array(val_buf), dtype=torch.float32)
    rew_tensor = torch.tensor(np.array(rew_buf), dtype=torch.float32)
    done_tensor = torch.tensor(np.array(done_buf), dtype=torch.float32)

    # Compute returns and advantages
    returns = compute_returns(rew_tensor, done_tensor, val_tensor, gamma, lam)
    adv = returns - val_tensor
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(10):  # PPO update iterations
        idx = np.random.permutation(len(obs_buf))
        for start in range(0, len(obs_buf), mini_batch_size):
            end = start + mini_batch_size
            mb_idx = idx[start:end]

            logits, values = policy(obs_tensor[mb_idx])
            dist = Normal(logits, torch.tensor(0.2))
            new_logp = dist.log_prob(act_tensor[mb_idx]).sum(axis=-1)
            ratio = (new_logp - logp_tensor[mb_idx]).exp()

            # PPO loss terms
            surrogate1 = ratio * adv[mb_idx]
            surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv[mb_idx]
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = ((returns[mb_idx] - values.squeeze()) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_reward = np.mean(rew_buf)
    reward_per_epoch.append(avg_reward)
    successes_per_epoch.append(success_count)

    scheduler.step()
    

    # Early stopping and best model tracking
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        best_model_state = policy.state_dict()
        epochs_without_improvement = 0
        print(f"🔥 New best model at epoch {epoch+1} with avg reward {avg_reward:.2f}")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    print(f"Epoch {epoch+1} completed. Avg reward: {avg_reward:.2f}")

# Plotting
plt.figure()
plt.plot(reward_per_epoch)
plt.title('Average reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.grid(True)
plt.show()

# Χρήση του καλύτερου μοντέλου από τη μνήμη
policy.load_state_dict(best_model_state)
policy.eval()
print("Loaded best model from memory for testing")


print("\nZero-shot generalization test on 5 unseen dynamics environments...")

successes = 0
test_runs = 10
final_positions = []
test_threshold = 0.3
#np.random.seed(42)

pygame.quit()
for i in range(test_runs):
    test_env = Point2DEnv()
    obs, _ = test_env.reset()

    # Randomized dynamics
    #test_env.mass = np.random.uniform(1.8, 2.5)
    #test_env.friction = np.random.uniform(0.25, 0.4)
    #test_env.sensor_noise = np.random.normal(0.1, 0.01, size=(2,))
    #test_env.position = np.array([0.0, 0.0])

    
    done = False
    steps = 0
    while not done and steps < test_env.max_steps:
        with torch.no_grad():
            logits, _ = policy(torch.tensor(obs, dtype=torch.float32))
            action = np.clip(logits.numpy(), test_env.action_space.low, test_env.action_space.high)
        obs, reward, terminated= test_env.step(action)
        done = terminated
        steps += 1
        test_env.render_3d()

    final_pos = test_env.position.copy()
    final_dist = np.linalg.norm(final_pos - test_env.goal)

    final_positions.append((final_pos.copy(), final_dist < test_threshold))  # True αν πέτυχε

    if final_dist < test_threshold and 'Success' not in plt.gca().get_legend_handles_labels()[1]:
        plt.scatter(*final_pos, color='blue', s=80, label="Success")
    elif final_dist >= test_threshold and 'Failure' not in plt.gca().get_legend_handles_labels()[1]:
        plt.scatter(*final_pos, color='red', s=80, label="Failure")


    plt.plot([0, final_pos[0]], [0, final_pos[1]], '--', color='gray', alpha=0.4)

    if final_dist < test_threshold:
        successes += 1
        print(f"✅ Test {i+1}: Success")
    else:
        print(f"❌ Test {i+1}: Failure (dist={final_dist:.3f}, steps={steps})")

    print(f"Final distance: {final_dist:.3f}")
    pygame.quit()


# === Σχεδίαση όλων των τελικών θέσεων ===
for pos, success in final_positions:
    color = 'blue' if success else 'red'
    label = "Success" if success else "Failure"

    if label not in plt.gca().get_legend_handles_labels()[1]:
        plt.scatter(*pos, color=color, s=80, label=label)
    else:
        plt.scatter(*pos, color=color, s=80)

    plt.plot([0, pos[0]], [0, pos[1]], '--', color='gray', alpha=0.4)


plt.show(block=True)

print(f"\nZero-shot generalization success rate: {successes}/{test_runs} ({100 * successes / test_runs:.1f}%)")


# Save the trained model
# if best_model_state is not None:
#     torch.save(best_model_state, "ppo_best.pt")
#     print("✅ Best model saved as 'ppo_best.pt'")
#Load the best model for testing
# policy.load_state_dict(torch.load("ppo_best.pt"))
# policy.eval()
# print("✅ Loaded best model for testing")
