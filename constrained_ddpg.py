# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# from stable_baselines3.common.buffers import ReplayBuffer
from custom_rb import CustomReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import safety_gymnasium
from omnisafe.common.lagrange import Lagrange

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    agent: str = "HalfCheetah"
    """the agent type"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    
    lag_tune: bool = False
    "to train a cost-constrained agent"
    lag_mult_init: float = 10.
    "initial value of the lagrangian multiplier"
    lag_update_freq: int = 100
    "the frequency of updating lagrangian multiplier"

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import stable_baselines3 as sb3

    args = tyro.cli(Args)
    run_name = f"{args.agent}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = safety_gymnasium.make(f"Safety{args.agent}Velocity-v1")

    actor = Actor(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)


    cf1 = QNetwork(env).to(device)
    cf1_target = QNetwork(env).to(device)
    cf1_target.load_state_dict(cf1.state_dict())
    c_optimizer = optim.Adam(list(cf1.parameters()), lr = args.learning_rate)
    
    lagrange = Lagrange(cost_limit = 1, lagrangian_multiplier_init = args.lag_mult_init, lambda_lr = args.learning_rate, lambda_optimizer = 'Adam')
    cf1_loss = torch.tensor([0.]) # DUMMY INIT REQUIRED FOR LOGGING
    cf1_a_values = torch.tensor(0.) # DUMMY INIT REQUIRED FOR LOGGING
    
    env.observation_space.dtype = np.float32
    
    rb = CustomReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    epi_return = 0
    epi_length = 0
    epi_cost = 0
    
    run_cost = 0
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(torch.Tensor(obs).to(device))
                action += torch.normal(0, actor.action_scale * args.exploration_noise)
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        
        epi_return += reward
        epi_length += 1
        epi_cost += cost
        
        run_cost += cost
        
        rb.add(obs, next_obs, action, reward, cost, terminated, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
        if terminated or truncated:
            obs, _ = env.reset()
            print(f"{global_step} - Return: {epi_return}; Cost: {epi_cost}; Length: {epi_length}")

            writer.add_scalar("charts/return", epi_return, global_step)
            writer.add_scalar("charts/length", epi_length, global_step)
            writer.add_scalar("charts/cost", epi_cost, global_step)
            
            writer.add_scalar("charts/lag_multiplier", lagrange.lagrangian_multiplier.item(), global_step)

            epi_return = 0
            epi_length = 0
            epi_cost = 0
            
            lagrange.update_lagrange_multiplier(run_cost)
            run_cost = 0            

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            
            if global_step % args.lag_update_freq == 0:
                lagrange.update_lagrange_multiplier(run_cost)
                run_cost = 0

            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()
            
            if args.lag_tune:
                with torch.no_grad():
                    cf1_next_target = cf1_target(data.next_observations, next_state_actions)
                    next_c_value = data.costs.flatten() + (1 - data.dones.flatten()) * args.gamma * (cf1_next_target).view(-1)

                cf1_a_values = cf1(data.observations, data.actions).view(-1)
                cf1_loss = F.mse_loss(cf1_a_values, next_c_value)

                # optimize the model
                c_optimizer.zero_grad()
                cf1_loss.backward()
                c_optimizer.step()


            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                if args.lag_tune:
                    c_loss = cf1(data.observations, actor(data.observations)).mean() * lagrange.lagrangian_multiplier.item()
                    actor_loss = (actor_loss + c_loss) / (1 + lagrange.lagrangian_multiplier.item())
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                for param, target_param in zip(cf1.parameters(), cf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/cf1_values", cf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/cf1_loss", cf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)


    env.close()
    writer.close()