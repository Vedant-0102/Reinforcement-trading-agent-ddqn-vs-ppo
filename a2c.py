import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium import spaces

# These helpers are included here so this demo does not need another project file.
def ask_value(question, default, convert=str):
    """Prompt until a valid value is entered; blank input selects the default."""
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return convert(raw)
        except ValueError:
            print("That value is not valid. Please try again.")


# The indicators summarize recent context; none of them use future prices.
def make_market_features(prices):
    """Build causal return, volatility, moving-average, and RSI features."""
    price = pd.Series(np.asarray(prices, dtype=float))
    log_return = np.log(price).diff()
    delta = price.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / (down + 1e-12))
    frame = pd.DataFrame({
        "return_1": log_return,
        "return_5": np.log(price).diff(5),
        "return_20": np.log(price).diff(20),
        "volatility_10": log_return.rolling(10).std(),
        "volatility_20": log_return.rolling(20).std(),
        "ma_gap_10": price / price.rolling(10).mean() - 1,
        "ma_gap_30": price / price.rolling(30).mean() - 1,
        "rsi_14": rsi / 100 - 0.5,
    })
    return frame.to_numpy(dtype=np.float32)


# Ask for the stock history used by this run; the algorithm learns only from this ticker.
def load_market_data():
    """Ask for one stock ticker and download its historical adjusted closing prices."""
    import yfinance as yf

    while True:
        ticker = input("Stock ticker symbol (for example AAPL): ").strip().upper()
        if not ticker:
            print("Please enter a ticker symbol.")
            continue
        period = ask_value("Historical period (for example 5y)", "5y")
        try:
            # Download adjusted prices so splits and dividends are handled consistently.
            frame = yf.download(ticker, period=period, auto_adjust=True,
                                progress=False, threads=False)
            if frame is None or frame.empty:
                raise ValueError(f"No historical prices were returned for {ticker}.")
            close = frame["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            prices = close.dropna().to_numpy(dtype=float)
            # A short period can still work; the run setup checks whether its train/test
            # split can support the observation window and episode length the user selects.
            if len(prices) < 12:
                raise ValueError(
                    f"Only {len(prices)} prices were returned; choose a period with at least 12 prices."
                )
            return prices, ticker
        except (OSError, ValueError, ImportError, KeyError) as error:
            print(f"Could not load {ticker}: {error}. Please try again.")
# This small simulator turns price history into observations, actions, and learning rewards.
class TradingEnvironment(gym.Env):
    """Simulate short, flat, and long positions with transaction-cost rewards."""
    def __init__(self, prices, features, window=10, cost=0.001,
                 random_start=False, episode_length=250, seed=0, initial_cash=10000.0):
        """Store market arrays and configure observation and action spaces."""
        super().__init__()
        self.prices = np.asarray(prices, dtype=float)
        self.features = np.asarray(features, dtype=np.float32)
        self.window = window
        self.cost = cost
        self.initial_cash = float(initial_cash)
        self.random_start = random_start
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-10, 10,
            shape=(window * self.features.shape[1] + 1,), dtype=np.float32)

    def _observation(self):
        """Return the recent feature window and the current position."""
        recent = self.features[self.t - self.window + 1:self.t + 1].reshape(-1)
        return np.r_[recent, self.position].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Start a full evaluation episode or a random training segment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        first = self.window - 1
        last = len(self.prices) - 2
        if self.random_start:
            high = max(first, last - self.episode_length)
            self.t = int(self.rng.integers(first, high + 1))
            self.t_end = min(last, self.t + self.episode_length)
        else:
            self.t, self.t_end = first, last
        self.position = 0
        self.portfolio_value = self.initial_cash
        self.equity_curve = [self.initial_cash]
        self.returns, self.positions, self.actions = [], [], []
        return self._observation(), {}

    def step(self, action):
        """Apply the chosen target position and return net reward and next observation."""
        position = int(action) - 1
        trade_size = abs(position - self.position)
        market_return = self.prices[self.t + 1] / self.prices[self.t] - 1
        net_return = position * market_return - self.cost * trade_size
        self.position = position
        self.returns.append(float(net_return))
        self.positions.append(position)
        self.actions.append(int(action))
        self.t += 1
        return self._observation(), float(net_return * 100), self.t >= self.t_end, False, {}


def _equity(returns, starting_cash=1.0):
    """Convert returns into a dollar-value portfolio curve starting from chosen cash."""
    return starting_cash * np.r_[1.0, np.cumprod(1 + np.asarray(returns, dtype=float))]


# Keep reporting separate from reward scaling so the displayed values remain interpretable.
def _report_metrics(returns, positions, starting_cash=10000.0):
    """Calculate presentation metrics from unscaled daily returns and positions."""
    returns = np.asarray(returns, dtype=float)
    equity = _equity(returns)
    peak = np.maximum.accumulate(equity)
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    volatility = returns.std()
    years = max(len(returns) / 252, 1e-9)
    annual_return = equity[-1] ** (1 / years) - 1
    drawdown = float(np.min(equity / peak - 1))
    position_array = np.asarray(positions)
    return {
        "total_return": float(equity[-1] - 1),
        "ending_cash": float(starting_cash * equity[-1]),
        "annual_return": float(annual_return),
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility else 0.0,
        "sortino": float(returns.mean() / downside * np.sqrt(252)) if downside else 0.0,
        "max_drawdown": drawdown,
        "turnover": float(np.abs(np.diff(np.r_[0, position_array])).sum()),
        "time_in_market": float(np.mean(position_array != 0)),
    }


# This is the complete run flow: ask, train on earlier data, score on later data, and save files.
def run_standalone_demo(algorithm_name, agent_factory, dqn=False):
    """Prompt for inputs, train this file's algorithm, evaluate it, and save a report."""
    print(f"\n{algorithm_name} standalone stock-trading demonstration")
    prices, data_name = load_market_data()
    window = ask_value("Observation window in days", 10, int)
    train_fraction = ask_value("Training fraction (0.5–0.9)", 0.7, float)
    if window < 1 or not 0.5 <= train_fraction <= 0.9:
        raise ValueError("The observation window must be positive and training fraction must be 0.5–0.9.")

    # Work out the maximum episode length from this ticker's available training rows.
    # This avoids rejecting short downloads just because a fixed default was too large.
    split = int(len(prices) * train_fraction)
    max_episode_length = split - window - 2
    if window >= len(prices) - 2 or max_episode_length < 1 or len(prices) - split <= 2:
        raise ValueError(
            "This history is too short for the selected observation window and a holdout period. "
            "Choose a smaller window or a longer history period."
        )

    episodes = ask_value("Training episodes", 30, int)
    default_episode_length = min(100, max_episode_length)
    episode_length = ask_value(
        f"Trading days in each training episode (max {max_episode_length})",
        default_episode_length, int
    )
    initial_cash = ask_value("Starting portfolio cash ($)", 10000.0, float)
    cost = ask_value("Transaction cost per position unit (0.001 = 0.1%)", 0.001, float)
    seed = ask_value("Random seed", 7, int)
    if (episodes < 1 or episode_length < 1 or episode_length > max_episode_length
            or initial_cash <= 0 or cost < 0):
        raise ValueError(
            f"Episodes and episode length must be positive, episode length cannot exceed {max_episode_length}; "
            "cash must be positive and transaction cost nonnegative."
        )
    steps = episodes * episode_length

    # The split stays chronological, and only training rows set the feature scaling.
    raw_features = make_market_features(prices)
    mean = np.nanmean(raw_features[:split], axis=0)
    scale = np.nanstd(raw_features[:split], axis=0) + 1e-8
    features = np.clip(np.nan_to_num((raw_features - mean) / scale), -5, 5).astype(np.float32)

    # Training episodes use random windows from the early period; their length is user controlled.
    train_env = TradingEnvironment(prices[:split], features[:split], window, cost,
                                   random_start=True, episode_length=episode_length,
                                   seed=seed, initial_cash=initial_cash)
    history_start = split - window + 1
    test_env = TradingEnvironment(prices[history_start:], features[history_start:], window, cost,
                                  initial_cash=initial_cash)
    observation_size = window * features.shape[1] + 1
    agent = agent_factory(train_env, observation_size, seed, steps)
    print(f"Training for {episodes} episode(s) × {episode_length} days = {steps} steps")
    if dqn:
        training_rewards = agent.train(train_env, steps, seed=seed)
    else:
        agent.learn(total_timesteps=steps)
        training_rewards = getattr(agent, "episode_rewards", [])
        summary = getattr(agent, "training_summary", {})
        if summary:
            print("Training summary:", summary)
    print(f"Completed training episodes with rewards recorded: {len(training_rewards)}")

    observation, _ = test_env.reset()
    done = False
    while not done:
        if dqn:
            action = agent.act(observation, greedy=True)
        else:
            action, _ = agent.predict(observation, deterministic=True)
            action = int(action)
        observation, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    # Use buy-and-hold as a plain reference over the same test-period price changes.
    benchmark_positions = np.ones(len(prices) - split - 1, dtype=float)
    benchmark_returns = prices[split + 1:] / prices[split:-1] - 1
    if len(benchmark_returns):
        benchmark_returns[0] -= cost  # Buy-and-hold pays to enter once, then holds.
    learned = _report_metrics(test_env.returns, test_env.positions, initial_cash)
    benchmark = _report_metrics(benchmark_returns, benchmark_positions, initial_cash)
    report = pd.DataFrame([learned, benchmark], index=[algorithm_name, "Buy and hold"])
    output = os.path.join("results", algorithm_name.lower().replace(" ", "_"))
    os.makedirs(output, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in data_name)
    csv_path = os.path.join(output, f"{safe_name}_metrics.csv")
    image_path = os.path.join(output, f"{safe_name}_presentation.png")
    report.to_csv(csv_path)

    # One glance shows the market, how learning rewards changed, portfolio value, and actions.
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(prices, color="#4c78a8", linewidth=1.3, label="Price")
    axes[0, 0].axvline(split, color="#777777", linestyle="--", label="Train/test split")
    test_steps = split + np.arange(len(test_env.actions))
    action_array = np.asarray(test_env.actions)
    buy_steps = test_steps[action_array == 2]
    sell_steps = test_steps[action_array == 0]
    axes[0, 0].scatter(buy_steps, prices[buy_steps], marker="^", color="#2ca02c", s=28, label="Buy")
    axes[0, 0].scatter(sell_steps, prices[sell_steps], marker="v", color="#d62728", s=28, label="Sell")
    axes[0, 0].set_title(f"{data_name}: price and test-period trades")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    if training_rewards:
        reward_array = np.asarray(training_rewards, dtype=float)
        axes[0, 1].plot(np.arange(1, len(reward_array) + 1), reward_array,
                        color="#f28e2b", alpha=0.55, label="Episode reward")
        smooth_window = min(5, len(reward_array))
        if smooth_window > 1:
            smoothed = np.convolve(reward_array, np.ones(smooth_window) / smooth_window, mode="valid")
            axes[0, 1].plot(np.arange(smooth_window, len(reward_array) + 1), smoothed,
                            color="#e15759", linewidth=2, label=f"{smooth_window}-episode average")
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "No complete episode rewards were recorded",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Training reward by episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total reward")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(test_env.equity_curve, label=algorithm_name, linewidth=2)
    axes[1, 0].plot(_equity(benchmark_returns, initial_cash), label="Buy and hold")
    axes[1, 0].axhline(initial_cash, color="#777777", linestyle=":", linewidth=1, label="Starting cash")
    axes[1, 0].set_title("Holdout account value")
    axes[1, 0].set_xlabel("Test trading day")
    axes[1, 0].set_ylabel("Portfolio value ($)")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    action_counts = [np.count_nonzero(action_array == action) for action in (1, 2, 0)]
    axes[1, 1].bar(["Hold", "Buy", "Sell"], action_counts,
                   color=["#9c9c9c", "#2ca02c", "#d62728"])
    axes[1, 1].set_title("Hold / buy / sell decisions")
    axes[1, 1].set_ylabel("Number of test days")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle(f"{algorithm_name} trading summary", fontsize=15)
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    # Also open the chart window for a live demo; the PNG is already saved for slides.
    plt.show()
    plt.close(figure)

    print("\n=== Unseen holdout results ===")
    display = report[["total_return", "ending_cash", "sharpe", "max_drawdown", "turnover", "time_in_market"]].copy()
    display["total_return"] *= 100
    display["max_drawdown"] *= 100
    display = display.rename(columns={"total_return": "return_%", "ending_cash": "ending_cash_$", "max_drawdown": "drawdown_%"})
    print(display.round(3).to_string())
    print("Metrics:", os.path.abspath(csv_path))
    print("Presentation plot:", os.path.abspath(image_path))
    print("Historical simulation only; results are not a forecast.")


# The actor chooses actions; the critic estimates how useful the current state is.
class ActorCritic(nn.Module):
    """Shared feature network with a policy head (actor) and value head (critic)."""

    def __init__(self, observation_size, action_count, hidden_size=128):
        """Create actor logits and a scalar state-value estimate from observations."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_count)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations):
        """Return action logits and the critic's value estimate for each observation."""
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


class A2CAgent:
    """Train an on-policy actor-critic using n-step returns and advantage updates."""

    def __init__(self, env, seed=0, learning_rate=3e-4, gamma=0.99,
                 value_coefficient=0.5, entropy_coefficient=0.01,
                 rollout_steps=5, max_grad_norm=0.5, hidden_size=128):
        """Initialize the policy/value network, optimizer, and A2C hyperparameters."""
        self.env = env
        self.seed = seed
        # Gamma controls how much the critic values later rewards.
        self.gamma = gamma
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.rollout_steps = rollout_steps
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.network = ActorCritic(
            int(np.prod(env.observation_space.shape)),
            env.action_space.n,
            hidden_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def predict(self, observation, deterministic=True):
        """Select an action from one observation; match the common SB3 predict interface."""
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            logits, _ = self.network(state)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        return np.asarray(action.item()), None

    def _update_from_rollout(self, states, actions, rewards, dones, values, next_values):
        """Compute n-step advantages and apply one actor, critic, and entropy update."""
        # Advantage asks whether each outcome beat the critic’s expectation.
        advantages = torch.zeros_like(rewards)
        running_advantage = torch.tensor(0.0, device=self.device)
        for index in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[index]
            td_error = rewards[index] + self.gamma * next_values[index] * nonterminal - values[index]
            running_advantage = td_error + self.gamma * nonterminal * running_advantage
            advantages[index] = running_advantage

        logits, predicted_values = self.network(states)
        distribution = Categorical(logits=logits)
        log_probabilities = distribution.log_prob(actions)
        # Positive advantage raises an action’s probability; negative advantage lowers it.
        policy_loss = -(log_probabilities * advantages.detach()).mean()
        # Train the critic toward the return estimate; the actor uses the advantage above.
        value_loss = nn.functional.mse_loss(predicted_values, (advantages + values).detach())
        # Entropy rewards a little variety so the actor does not become certain too early.
        entropy_bonus = distribution.entropy().mean()
        loss = (policy_loss + self.value_coefficient * value_loss
                - self.entropy_coefficient * entropy_bonus)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(policy_loss.item()), float(value_loss.item()), float(entropy_bonus.item())

    def learn(self, total_timesteps):
        """Collect on-policy rollouts, update actor and critic, and return training summaries."""
        observation, _ = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_rewards = []
        updates = []
        steps = 0

        # A2C gathers a short rollout, updates both networks once, and then gathers again.
        while steps < total_timesteps:
            states, actions, rewards, dones, values, next_values = [], [], [], [], [], []
            for _ in range(min(self.rollout_steps, total_timesteps - steps)):
                state_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits, value = self.network(state_tensor.unsqueeze(0))
                    action_tensor = Categorical(logits=logits).sample()
                action = int(action_tensor.item())
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_tensor = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _, next_value = self.network(next_tensor.unsqueeze(0))

                states.append(state_tensor)
                actions.append(action_tensor.squeeze(0))
                rewards.append(float(reward))
                dones.append(float(done))
                values.append(value.squeeze(0))
                next_values.append(next_value.squeeze(0))
                episode_reward += float(reward)
                steps += 1

                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    next_observation, _ = self.env.reset()
                observation = next_observation

            update = self._update_from_rollout(
                torch.stack(states), torch.stack(actions),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device),
                torch.stack(values), torch.stack(next_values),
            )
            updates.append(update)

        self.episode_rewards = episode_rewards
        self.training_summary = {
            "timesteps": steps,
            "episodes": len(episode_rewards),
            "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else None,
            "mean_policy_loss": float(np.mean([u[0] for u in updates])),
            "mean_value_loss": float(np.mean([u[1] for u in updates])),
            "mean_entropy": float(np.mean([u[2] for u in updates])),
        }
        return self


def build_agent(env, obs_dim, seed, steps):
    """Build this standalone A2C learner for the selected environment."""
    return A2CAgent(env, seed=seed)

if __name__ == "__main__":
    run_standalone_demo("A2C", build_agent)
