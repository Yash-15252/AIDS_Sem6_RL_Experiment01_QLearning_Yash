# AIDS_Sem8_RL_Experiment01_QTaxiLearning

## Aim
    Implement Q-Learning agent to solve Taxi-v3 environment (pickup/drop passengers optimally).

## Problem Statement
Train taxi agent in 5x5 gridworld to pick up passenger from blue loc, drop at yellow loc safely.

## Brief Theory
Q-Learning: Off-policy TD method. Update: Q(s,a) ← Q(s,a) + α[r + γ maxQ(s',a') - Q(s,a)]. Table [500 states × 6 actions].

## Implementation Explanation
`src/RL_EXP_1.ipynb`:
- Load Gymnasium Taxi-v3 (render rgb_array)
- Q-table zeros [500,6]
- Random baseline, single episode demo
- Train 2000 episodes (α=0.618)
- Test optimal policy, render

## Results
After 2000 eps: Total reward ~7 (vs random 200+ steps). Screenshots: states 452, env renders, Q-update plot.
Add run screenshots to `screenshots/`.

## Conclusion
Q-Learning converged to optimal policy efficiently for tabular Taxi env.

## References
- Gymnasium Taxi-v3 docs
- Sutton&Barto RL book Ch6
