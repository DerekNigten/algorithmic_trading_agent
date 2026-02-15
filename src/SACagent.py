

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import matplotlib.pyplot as plt

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# Environment
# ============================================================================

class TradingEnv(gym.Env):
    def __init__(self, data_df, pred_arrays, max_shares=100, realistic_execution=False):
        super().__init__()
        
        # Store data
        self.data_df = data_df.reset_index(drop=True)
        
        # Calculate mid_price from BID_PRICE_1 and ASK_PRICE_1
        self.data_df['mid_price'] = (self.data_df['BID_PRICE_1'] + self.data_df['ASK_PRICE_1']) / 2
        
        # Store prediction arrays
        self.preds = pred_arrays
        
        self.max_shares = max_shares
        self.realistic_execution = realistic_execution
        self.n_steps = len(self.data_df) - 1
        
        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        self.step_idx = 0
        self.position = 0.0
        self.shares = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0  # Training-style PnL (for state)
        self.total_pnl_realistic = 0.0  # Realistic PnL (for reporting)
        self.pnl_history = []
        self.trade_count = 0
        return self._get_state()
    
    def _get_state(self):
        pred_features = np.concatenate([p[self.step_idx] for p in self.preds])
        mid_price = self.data_df.iloc[self.step_idx]['mid_price']
        
        # CRITICAL: Always use training-style PnL for state consistency
        state = np.concatenate([
            pred_features,
            [mid_price],
            [self.position, self.total_pnl]  # Training PnL, not realistic
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        trade_score = action[0]
        target_position = action[1]
        
        old_position = self.position
        old_shares = self.shares
        old_mid = self.data_df.iloc[self.step_idx]['mid_price']
        old_bid = self.data_df.iloc[self.step_idx]['BID_PRICE_1']
        old_ask = self.data_df.iloc[self.step_idx]['ASK_PRICE_1']
        
        # Move to next step
        self.step_idx += 1
        new_mid = self.data_df.iloc[self.step_idx]['mid_price']
        
        # Decide new position
        if trade_score > 0:
            new_position = np.clip(target_position, -1, 1)
        else:
            new_position = old_position
        
        # Calculate TRAINING reward (old style - free trading)
        price_change = new_mid - old_mid
        pnl_from_holding = old_shares * price_change
        reward_training = pnl_from_holding
        
        # Execute realistic bid/ask trade
        if trade_score > 0 and new_position != old_position:
            new_shares = new_position * self.max_shares
            shares_to_trade = new_shares - old_shares
            
            if shares_to_trade > 0:  # Buying
                self.cash -= shares_to_trade * old_ask
                self.trade_count += 1
            elif shares_to_trade < 0:  # Selling
                self.cash += abs(shares_to_trade) * old_bid
                self.trade_count += 1
            
            self.shares = new_shares
            self.position = new_position
        
        # Calculate REALISTIC reward (with bid/ask costs)
        old_portfolio = self.cash + old_shares * old_mid
        new_portfolio = self.cash + self.shares * new_mid
        reward_realistic = new_portfolio - old_portfolio
        
        # CRITICAL FIX: Always use training PnL for state consistency
        self.total_pnl += reward_training
        
        # Track realistic PnL separately for reporting
        self.total_pnl_realistic += reward_realistic
        self.pnl_history.append(self.total_pnl_realistic)
        
        # Return different reward based on mode, but state stays consistent
        if self.realistic_execution:
            reward = reward_realistic
        else:
            reward = reward_training
        
        done = self.step_idx >= self.n_steps - 1
        
        return self._get_state(), reward, done, {}


# ============================================================================
# SAC Agent
# ============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(1_000_000)
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor(state)
            return torch.tanh(mean).detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().numpy()[0]
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target, source in [(self.target_critic1, self.critic1), 
                               (self.target_critic2, self.critic2)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ============================================================================
# Data Loading
# ============================================================================

def load_and_split_data(data_path, pred_paths, train_ratio=0.8):
    print("Loading test_data.csv...")
    data_df = pd.read_csv(data_path)
    print(f"Data shape: {data_df.shape}")
    
    print("\nLoading predictions...")
    preds = []
    for i, path in enumerate(pred_paths):
        df = pd.read_csv(path)
        prob_cols = [c for c in df.columns if c.startswith('prob_')]
        if len(prob_cols) != 3:
            raise ValueError(f"Expected 3 prob columns in {path}, found: {prob_cols}")
        prob_cols_sorted = sorted(prob_cols)
        preds.append(df[prob_cols_sorted].values)
        print(f"  Model {i+1}: shape {preds[-1].shape}")
    
    min_length = min(len(data_df), *[len(p) for p in preds])
    print(f"\nAligning to minimum length: {min_length}")
    
    data_df = data_df.iloc[:min_length]
    preds = [p[:min_length] for p in preds]
    
    split_idx = int(min_length * train_ratio)
    
    train_data = data_df.iloc[:split_idx]
    val_data = data_df.iloc[split_idx:]
    
    train_preds = [p[:split_idx] for p in preds]
    val_preds = [p[split_idx:] for p in preds]
    
    print(f"\nTrain set: {len(train_data)} events")
    print(f"Val set: {len(val_data)} events")
    
    return train_data, train_preds, val_data, val_preds


# ============================================================================
# Training
# ============================================================================

def train():
    data_path = '/Users/derek/Desktop/hft_gitlab/data/test_data.csv'
    pred_paths = [
        '/Users/derek/Desktop/hft_gitlab/outputs/lstm/test_predictions.csv',
        '/Users/derek/Desktop/hft_gitlab/outputs/tcn/test_predictions.csv',
        '/Users/derek/Desktop/hft_gitlab/outputs/xgboost/xgb_test_predictions.csv',
        '/Users/derek/Desktop/hft_gitlab/outputs/transformer/transformer_test_predictions.csv'
    ]
    
    train_data, train_preds, val_data, val_preds = load_and_split_data(
        data_path, pred_paths, train_ratio=0.8
    )
    
    # Training: realistic_execution=False (training rewards)
    # Validation: realistic_execution=True (realistic rewards, but state stays consistent)
    train_env = TradingEnv(train_data, train_preds, max_shares=100, realistic_execution=False)
    val_env = TradingEnv(val_data, val_preds, max_shares=100, realistic_execution=True)
    
    agent = SACAgent(state_dim=15, action_dim=2)
    
    n_episodes = 5
    batch_size = 256
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    for episode in range(n_episodes):
        state = train_env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > batch_size and step_count % 10 == 0:
                agent.update(batch_size)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if step_count % 100000 == 0:
                print(f"  Step {step_count}/{train_env.n_steps}, Training Reward: ${episode_reward:.2f}, Realistic PnL: ${train_env.total_pnl_realistic:.2f}, Trades: {train_env.trade_count}")
        
        print(f"Episode {episode+1}/{n_episodes} - Training Reward: ${train_env.total_pnl:.2f}, Realistic PnL: ${train_env.total_pnl_realistic:.2f}, Trades: {train_env.trade_count}\n")
    
    # Validation
    print("="*60)
    print("VALIDATION")
    print("="*60)
    state = val_env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, _ = val_env.step(action)
    
    print(f"Validation PnL (realistic): ${val_env.total_pnl_realistic:.2f}")
    print(f"Validation Trades: {val_env.trade_count}")
    print(f"Trade Frequency: {val_env.trade_count/val_env.n_steps*100:.1f}%")
    print(f"Final Position: {val_env.position:.2f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(val_env.pnl_history, linewidth=1.5, color='steelblue')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cumulative PnL ($) [with bid/ask costs]', fontsize=12)
    plt.title(f'Validation PnL (Realistic) - Final: ${val_env.total_pnl_realistic:.2f}, Trades: {val_env.trade_count}', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_pnl_realistic.png', dpi=150)
    print("\n✓ PnL plot saved to validation_pnl_realistic.png")
    
    return agent, train_env, val_env


if __name__ == "__main__":
    agent, train_env, val_env = train()