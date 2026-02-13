import torch
import torch.nn.functional as F

from meta_heuristics.RL.dr_alns_env import DRALNSEnv
from meta_heuristics.RL.dr_alns_policy import DRALNSPolicy
from meta_heuristics.problem_instance import ProblemInstance

device = torch.device("cpu")
torch.set_num_threads(1)


"""
Interact with the environment to collect a trajectory of 
    - states
    - actions
    - rewards
    - log probabilities
    - values
"""
def collect_trajectory(env, policy, horizon):
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    # Get initial state
    state = env.reset()

    for _ in range(horizon):
        # Get logits(actions to be taken)/value from policy
        logits, value = policy(state)
        # Turn into probabilities
        probs = torch.softmax(logits, dim=-1)

        # Pick an action at random according to the probabilities given by the network.
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # TODO: maybe explain better?
        # Log probability for PPO
        log_prob = dist.log_prob(action)

        # Collect all parameters
        next_state, reward, done = env.step(action.item())

        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        log_probs.append(log_prob)
        # Squeeze: remove extra unnecessary dimension from the value head (e.g. [[value]] -> [value])
        values.append(value.squeeze())

        state = next_state
        if done:
            break

    return states, actions, rewards, log_probs, values

"""
Compute discounted cumulative reward for each time step in the trajectory.
gamma = discount factor
"""
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


"""
Update PPO
"""
def ppo_update(policy, optimizer,
               states, actions, old_log_probs,
               returns, values,
               clip_eps=0.2):

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    # Compute new logits, values, and log probabilities
    logits, new_values = policy(states)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    # Compute advantage
    new_log_probs = dist.log_prob(actions)
    advantages = returns - new_values.squeeze().detach() #TODO explain detach? something about

    # Compute policy ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    # Clip it to epsilon to stabilize updates
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    # Compute policy loss
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # Compute value loss (MSE between predicted value and return)
    value_loss = F.mse_loss(new_values.squeeze(), returns)
    loss = policy_loss + 0.5 * value_loss

    # Clear gradients (to 0) before backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_dralns(problem_instance,
                 num_destroy_ops,
                 max_env_steps=200,
                 rollout_horizon=100,
                 epochs=500):

    env = DRALNSEnv(problem_instance, max_env_steps)
    policy = DRALNSPolicy(state_dim=7, num_actions=num_destroy_ops).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    for epoch in range(epochs):
        states, actions, rewards, log_probs, values = collect_trajectory(
            env, policy, rollout_horizon
        )

        returns = compute_returns(rewards)

        ppo_update(
            policy, optimizer,
            states, actions, log_probs,
            returns, values
        )

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Total reward: {sum(r.item() for r in rewards):6.2f} | "
                f"Best violations: {env.best_violations}"
            )

    return policy


if __name__ == '__main__':
    problem = ProblemInstance(num_groups=3, num_players=9, num_weeks=4)
    train_dralns(problem, 3)


