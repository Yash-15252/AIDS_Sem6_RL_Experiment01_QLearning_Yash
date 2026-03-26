# AIDS_Sem8_RL_Experiment01_Taxi

## ***YASH KHAMKAR - 221A030***
## ***Q-Learning on Taxi-v3 Environment***

---

## Aim
Train Q-learning agent on **Taxi-v3** Gymnasium environment to learn optimal pickup/dropoff policy.

## Problem Statement
```
Taxi-v3: 5×5 grid taxi world (500 states)
• State encoding: taxi_loc(25) × pass_loc(4) × dest(4) = 500 states
• Actions (6): 0=↓, 1=↑, 2=→, 3=←, 4=pickup, 5=dropoff  
• Rewards: +20 dropoff, -1 move, -10 illegal pickup/dropoff
• Goal: Maximize expected reward (optimal policy solves in ~8-12 steps)
```

**Observation Space**: Discrete(500), **Action Space**: Discrete(6)

## Brief Theory
**Tabular Q-Learning** (Off-policy TD):
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
π(s) = argmax_a Q(s,a)    (greedy policy)
```

**Parameters**: α=0.618 (decaying), γ=1.0 implicit, 2000 episodes.

## Implementation Explanation
`RL_EXP_1.ipynb` implements:

```
1. Environment Setup: gym.make('Taxi-v3', render_mode='rgb_array')
2. Q-table: 500×6 zeros array
3. Random Baseline: ~200+ steps, reward ~-200
4. Single Episode Demo: Q-learning update visualization  
5. Full Training: 2000 episodes until reward=+7 (optimal)
6. Test: argmax Q policy → solves in minimal steps
7. Render: Visual taxi navigation (pickup→dropoff)
```

**Key Steps**:
```
• Q[state, action] += α (r + max Q(s') - Q(s,a))
• Exploit: action = argmax Q[state]
• Terminal: reward==+20 (successful dropoff)
```

## Results
```
Random Agent: ~200 steps, reward ~-200
After 1 episode: Improved but suboptimal
After 2000 episodes: Reward=+7 (optimal policy)
Test Run: Taxi picks up → navigates → drops off (+20)
```

**Convergence**: Q-learning discovers optimal policy π* maximizing +20 reward

## Sample Output
```
Episode 100 Total Reward: -X
Episode 2000 Total Reward: +7 ✓
Final Render: [Visual: Taxi reaches dest with passenger]
```

## Conclusion
 **Q-Learning Success**: Tabular method solves Taxi-v3 optimally  
 **Convergence**: 2000 episodes sufficient for 500×6 Q-table  
 **Reward Shaping**: +20 goal, -1/-10 penalties guide learning  
 **Visualization**: Render confirms correct pickup/dropoff sequence  
 **Foundation**: Model-free tabular RL baseline  

**Performance**: Optimal policy: 8-12 steps to +20 reward.

## References
1. Sutton & Barto, "RL: An Introduction" (Ch. 6: Q-Learning)
2. Gymnasium: Taxi-v3 Documentation
3. AIDS Sem8 RL Course

## Setup & Run
```bash
cd AIDS_Sem8_RL_Experiment01_Taxi
pip install -r requirements.txt
jupyter notebook RL_EXP_1.ipynb
```

**Requirements**:
```
gymnasium
numpy
```

---

