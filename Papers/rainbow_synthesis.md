# DQN Improvements shown in Rainbow DQN Paper

### Double Q-Learning:
The max operator in standard Q-learning and DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To prevent this, we can decouple the selection from the evaluation. Hence changing the target (Y) and keeping the rest of the DQN algorithm intact:
Y = Rt+1 + γ max a Q(St+1, a; θ’)  to Y  = Rt+1 + γQ(St+1, argmax a Q(St+1, a; θ); θ’ ).
Where θ’ is the target network which continues to be periodically updated as in regular DQN.
This approach stabilizes learning and avoid falling into suboptimal policies, outperforming regular DQN.

### Prioritized Experience Replay
Framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. Replaying transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error, increases learning.
The central component of prioritized replay is the criterion by which the importance of each transition is measured. When choosing TD error as the criterion a problem of diversity is encountered leading to some transitions not to be repeated at all. To overcome these issues, we introduce a stochastic sampling method that interpolates between pure greedy prioritization and uniform random sampling.

### Dueling network architecture (not very useful)
The idea is to separate the Q value prediction into value function and advantage due to the fact that in many states it is not important to know the value of taking every different action as they do not produce any change in the environment.
In order to produce this, the CNN is modified in order to output the value function on one stream and the advantage on the order.

### Multi-Step Reinforcement Learning: A Unifying Algorithm

### A Distributional Perspective on Reinforcement Learning
Uses value distribution instead of a value function.
