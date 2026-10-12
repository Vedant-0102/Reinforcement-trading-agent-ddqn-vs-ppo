"""Run Double DQN and PPO with the same stock data and settings for comparison."""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import double_dqn as ddqn
import ppo


def _evaluate(agent, env, algorithm):
    observation, _ = env.reset()
    done = False
    while not done:
        if algorithm == "Double DQN":
            action = agent.act(observation, greedy=True)
        else:
            action, _ = agent.predict(observation, deterministic=True)
            action = int(action)
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated


def main():
    print("\nDouble DQN vs PPO stock-trading comparison")
    prices, ticker = ddqn.load_market_data()
    window = ddqn.ask_value("Observation window in days", 10, int)
    train_fraction = ddqn.ask_value("Training fraction (0.5–0.9)", 0.7, float)
    split = int(len(prices) * train_fraction)
    max_episode_length = split - window - 2
    if window < 1 or not 0.5 <= train_fraction <= 0.9:
        raise ValueError("The observation window must be positive and training fraction 0.5–0.9.")
    if window >= len(prices) - 2 or max_episode_length < 1 or len(prices) - split <= 2:
        raise ValueError("This history is too short for the selected window and a holdout period.")

    episodes = ddqn.ask_value("Training episodes", 30, int)
    episode_length = ddqn.ask_value(
        f"Trading days in each training episode (max {max_episode_length})",
        min(100, max_episode_length), int,
    )
    initial_cash = ddqn.ask_value("Starting portfolio cash ($)", 10000.0, float)
    cost = ddqn.ask_value("Transaction cost per position unit (0.001 = 0.1%)", 0.001, float)
    seed = ddqn.ask_value("Random seed", 7, int)
    if (episodes < 1 or episode_length < 1 or episode_length > max_episode_length
            or initial_cash <= 0 or cost < 0):
        raise ValueError("Episodes and episode length must be positive and within the displayed maximum; "
                         "cash must be positive and transaction cost nonnegative.")
    steps = episodes * episode_length

    raw_features = ddqn.make_market_features(prices)
    mean = np.nanmean(raw_features[:split], axis=0)
    scale = np.nanstd(raw_features[:split], axis=0) + 1e-8
    features = np.clip(np.nan_to_num((raw_features - mean) / scale), -5, 5).astype(np.float32)
    history_start = split - window + 1
    obs_dim = window * features.shape[1] + 1

    def make_train_env():
        return ddqn.TradingEnvironment(
            prices[:split], features[:split], window, cost, random_start=True,
            episode_length=episode_length, seed=seed, initial_cash=initial_cash,
        )

    def make_test_env():
        return ddqn.TradingEnvironment(
            prices[history_start:], features[history_start:], window, cost,
            initial_cash=initial_cash,
        )

    print(f"\nShared settings: {ticker}, {steps} training steps per algorithm, seed {seed}")
    print("Training Double DQN...")
    dqn_env = make_train_env()
    dqn_agent = ddqn.build_agent(dqn_env, obs_dim, seed, steps)
    dqn_rewards = dqn_agent.train(dqn_env, steps, seed=seed)
    dqn_test = make_test_env()
    _evaluate(dqn_agent, dqn_test, "Double DQN")

    print("Training PPO...")
    ppo_env = make_train_env()
    ppo_agent = ppo.build_agent(ppo_env, obs_dim, seed, steps)
    ppo_agent.learn(total_timesteps=steps)
    ppo_rewards = ppo_agent.episode_rewards
    ppo_test = make_test_env()
    _evaluate(ppo_agent, ppo_test, "PPO")

    benchmark_positions = np.ones(len(prices) - split - 1, dtype=float)
    benchmark_returns = prices[split + 1:] / prices[split:-1] - 1
    if len(benchmark_returns):
        benchmark_returns[0] -= cost
    rows = {
        "Double DQN": ddqn._report_metrics(dqn_test.returns, dqn_test.positions, initial_cash),
        "PPO": ddqn._report_metrics(ppo_test.returns, ppo_test.positions, initial_cash),
        "Buy and hold": ddqn._report_metrics(benchmark_returns, benchmark_positions, initial_cash),
    }
    report = pd.DataFrame.from_dict(rows, orient="index")

    output = os.path.join("results", "double_dqn_vs_ppo")
    os.makedirs(output, exist_ok=True)
    report_path = os.path.join(output, f"{ticker}_comparison_metrics.csv")
    image_path = os.path.join(output, f"{ticker}_comparison.png")
    try:
        report.to_csv(report_path)
    except PermissionError:
        # Windows may lock a previous result while it is open in a spreadsheet app.
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output, f"{ticker}_comparison_metrics_{suffix}.csv")
        image_path = os.path.join(output, f"{ticker}_comparison_{suffix}.png")
        report.to_csv(report_path)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(prices, color="#4c78a8", label="Price")
    axes[0, 0].axvline(split, color="#777777", linestyle="--", label="Train/test split")
    for env, name, color in ((dqn_test, "Double DQN", "#2ca02c"), (ppo_test, "PPO", "#d62728")):
        actions = np.asarray(env.actions)
        test_steps = split + np.arange(len(actions))
        buys, sells = test_steps[actions == 2], test_steps[actions == 0]
        axes[0, 0].scatter(buys, prices[buys], marker="^", s=22, alpha=0.7,
                           color=color, label=f"{name} buy")
        axes[0, 0].scatter(sells, prices[sells], marker="v", s=22, alpha=0.7,
                           color=color, label=f"{name} sell")
    axes[0, 0].set_title(f"{ticker}: holdout trades")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=0.2)

    if dqn_rewards:
        axes[0, 1].plot(dqn_rewards, label="Double DQN", color="#2ca02c")
    if ppo_rewards:
        axes[0, 1].plot(ppo_rewards, label="PPO", color="#d62728")
    axes[0, 1].set_title("Training reward by completed episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total reward")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    for env, name, color in ((dqn_test, "Double DQN", "#2ca02c"),
                             (ppo_test, "PPO", "#d62728")):
        axes[1, 0].plot(ddqn._equity(env.returns, initial_cash), label=name, color=color)
    axes[1, 0].plot(ddqn._equity(benchmark_returns, initial_cash),
                    label="Buy and hold", color="#4c78a8")
    axes[1, 0].axhline(initial_cash, color="#777777", linestyle=":", label="Starting cash")
    axes[1, 0].set_title("Holdout account value")
    axes[1, 0].set_xlabel("Test trading day")
    axes[1, 0].set_ylabel("Portfolio value ($)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    labels, counts, colors = [], [], []
    for env, name, color in ((dqn_test, "Double DQN", "#2ca02c"),
                             (ppo_test, "PPO", "#d62728")):
        actions = np.asarray(env.actions)
        for action, decision in ((1, "Hold"), (2, "Buy"), (0, "Sell")):
            labels.append(f"{name}\n{decision}")
            counts.append(int(np.count_nonzero(actions == action)))
            colors.append(color)
    axes[1, 1].bar(labels, counts, color=colors)
    axes[1, 1].set_title("Hold / buy / sell decisions")
    axes[1, 1].set_ylabel("Number of test days")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle(f"{ticker}: Double DQN vs PPO", fontsize=15)
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.show()
    plt.close(figure)

    display = report.copy()
    display["total_return"] *= 100
    display["max_drawdown"] *= 100
    display = display.rename(columns={
        "total_return": "return_%", "max_drawdown": "drawdown_%",
        "ending_cash": "ending_cash_$",
    })
    print("\n=== Holdout comparison ===")
    print(display.round(3).to_string())
    print("\nHold / buy / sell days:")
    for name, env in (("Double DQN", dqn_test), ("PPO", ppo_test)):
        actions = np.asarray(env.actions)
        print(f"{name}: hold={np.count_nonzero(actions == 1)}, "
              f"buy={np.count_nonzero(actions == 2)}, sell={np.count_nonzero(actions == 0)}")
    print("Metrics:", os.path.abspath(report_path))
    print("Comparison chart:", os.path.abspath(image_path))
    print("Historical simulation only; results are not a forecast.")


if __name__ == "__main__":
    main()
