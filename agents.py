"""Custom DQN family: vanilla DQN, Double DQN, Dueling DQN, Double+Dueling.
Fixes vs v1: Huber loss, grad clipping, epsilon decayed linearly over env steps
(not per gradient step), target net updated on a fixed schedule, seedable."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.adv = nn.Linear(hidden, n_actions)
        else:
            self.q = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.body(x)
        if self.dueling:
            a = self.adv(h)
            return self.value(h) + a - a.mean(dim=1, keepdim=True)
        return self.q(h)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)

    def push(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.d[self.i] = s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.n


class DQNAgent:
    def __init__(self, obs_dim, n_actions=3, double=False, dueling=False, lr=5e-4,
                 gamma=0.99, buffer_size=50_000, batch_size=64, hidden=128,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000,
                 target_update=500, learning_starts=1_000, seed=0, device=None):
        self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma, self.batch_size = n_actions, gamma, batch_size
        self.double = double
        self.policy = QNet(obs_dim, n_actions, hidden, dueling).to(self.device)
        self.target = QNet(obs_dim, n_actions, hidden, dueling).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.target_update, self.learning_starts = target_update, learning_starts
        self.steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def _learn(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        s, s2 = (torch.as_tensor(x, device=self.device) for x in (s, s2))
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:  # action chosen by policy net, evaluated by target net
                a2 = self.policy(s2).argmax(dim=1, keepdim=True)
                q2 = self.target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.target(s2).max(dim=1)[0]
            y = r + self.gamma * q2 * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def train(self, env, total_steps, seed=None, log_every=0):
        obs, _ = env.reset(seed=seed if seed is not None else self.seed)
        ep_rew, ep_rewards = 0.0, []
        for step in range(1, total_steps + 1):
            a = self.act(obs)
            obs2, r, term, trunc, _ = env.step(a)
            self.buffer.push(obs, a, r, obs2, float(term))
            obs, ep_rew = obs2, ep_rew + r
            self.steps += 1
            if len(self.buffer) >= max(self.learning_starts, self.batch_size):
                self._learn()
                if self.steps % self.target_update == 0:
                    self.target.load_state_dict(self.policy.state_dict())
            if term or trunc:
                ep_rewards.append(ep_rew)
                ep_rew = 0.0
                obs, _ = env.reset()
                if log_every and len(ep_rewards) % log_every == 0:
                    print(f"    step {step}/{total_steps} eps={self.epsilon:.2f} "
                          f"mean_ep_reward={np.mean(ep_rewards[-log_every:]):.3f}")
        return ep_rewards
