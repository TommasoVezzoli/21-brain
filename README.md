# 21-brain: Learning Blackjack with Double Q-Learning and Card Counting

## 🧠 Overview

This project explores reinforcement learning (RL) applied to the game of Blackjack. We develop an OpenAI Gym-compatible Blackjack environment and train intelligent agents using **Double Q-Learning**, incorporating complex decision mechanics like **doubling**, **splitting**, and **card counting** (Hi-Lo system).

Our goal is to answer:  
**Can Double Q-Learning effectively learn Blackjack strategies, including betting and count-based adaptations?**

## 🎯 Key Features

- 🃏 Custom Gym environment implementing full Blackjack rules (hit, stand, double, split, betting)
- 🔢 Card counting support using **true count** (Hi-Lo)
- 🤖 Double Q-Learning agents trained over four progressive phases
- 📈 Performance evaluated against the Basic Strategy with visual heatmaps and metrics

## 🛠️ Environment

The Blackjack simulation supports:

- Variable number of decks and reshuffling threshold
- Full action space: `hit`, `stand`, `double`, `split`, and bet sizing
- Card counting using true count as part of the observation space

### Observation Space

- Player score (1–31)
- Dealer upcard (1–10)
- Soft hand indicator (1 if soft hand, 0 otherwise)
- Indicator for ability to split
- Pair value (if applicable)
- True count (normalized by decks remaining)

### Action Space

1. **Betting Phase**: Select bet size based on true count  
2. **Gameplay Phase**: Choose among `hit`, `stand`, `double`, `split`

### Rewards

- Win: +1  
- Loss: -1  
- Natural Blackjack: +1.5  
- Rewards from `double` or `split` scale according to outcome

## 📚 Training Pipeline

Training is divided into 4 stages, each implemented in its own Jupyter notebook:

1. `dq_hit_stand.ipynb`: Learn `hit` and `stand` actions
2. `dq_hit_stand_dd.ipynb`: Add support for `double down`
3. `betting.ipynb`: Train dynamic betting policy using card count
4. `split.ipynb`: Add support for `split` and multi-hand logic

Each stage builds upon the previous Q-table and training knowledge to incrementally improve the policy.

## 📊 Results

| Strategy             | Win Rate | Draw Rate | Loss Rate | Avg. Reward |
|----------------------|----------|-----------|-----------|-------------|
| Stage 1: Hit/Stand   | 43.2%    | 8.7%      | 48.1%     | -0.0258     |
| Stage 2: +Double     | 43.2%    | 8.6%      | 48.2%     | -0.0092     |
| Stage 3: +Betting    | 43.3%    | 8.6%      | 48.1%     | -0.0044     |
| Stage 4: +Split      | 43.4%    | 8.9%      | 47.7%     | **+0.0148** |

Agents learn to closely follow the Basic Strategy, adjust bets according to favorable deck conditions, and achieve positive returns in later stages.

## 📁 Repository Structure

```
21-brain/
├── src/
│   ├── env.py                 # Custom Gym environment
│   ├── utils.py               # Helper functions for training 
│   └── blackjack.py           # Setup classes for blackjack game
├── notebooks/
│   ├── dq_hit_stand.ipynb     # Stage 1: Hit/Stand training
│   ├── dq_hit_stand_dd.ipynb  # Stage 2: Add Double Down
│   ├── betting.ipynb          # Stage 3: Betting strategy
│   └── split.ipynb            # Stage 4: Add Split logic
├── strategies/                # Learnt strategies in CSV files 
├── README.md
├── 21_brain.pdf               # Report of the project                   
├── requirements.txt           
└── project_rules.pdf          
```


## 🧪 Future Work

- Multi-player Blackjack environment
- Finer-grained or continuous bet sizing
- Curriculum-based training (simple to complex hands)
- Deep Q-Networks (DQN) for large state generalization
