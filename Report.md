[//]: # (Image References)

[image1]: https://github.com/dariocazzani/banaNavigation/blob/master/images/DQN-equation.png
[image2]: https://github.com/dariocazzani/banaNavigation/blob/master/images/scores.png

# Report

## Learning Algorithm

### Overview:
* In the Reinforcement Learning Framework, the goal of the agent is to maximize
the expected cumulative reward.
* The agent makes a decision (which action to take), based on the current state and the
reward that the previous action led to.
* At first decision are made at random. the **Q-table** (a table that maps _state-action_ pairs
to an expected return), is initialized at random.
* **SARSAMAX** or **Q-learning** is an update algorithm that let's the agent improve
the **Q-table** as it interacts with the environment, and eventually converging to
an **optimal policy**.
* A policy is a function that maps any state to an action.
* When the state space is continuous or it is too big to build a Q-table, it is
possible to learn an approximation of such space with a **Non linear function approximator**
* The algorithm used for this problem is **Deep Q-learning**.

### Deep Q-learning
* The basic idea of Deep Q-learning is to use **Deep Neural Networks** as function approximators.
* The output of a Deep-QN, unlike in standard RL, is a distribution over all possible actions.
* Similarly to Q-learning, Deep Q-learning is an online learning algorithm. This
means that the agent improves at the task while interacting with the environment.
* In order to break potential harmful **correlations** between consecutive states, agents are
provided with a memory (**Replay Buffer**) where collections of [`State`, `Action`, `Reward`, `Next State`]
are stored. At learning time, a batch of such tuples are selected to run a learning step
and update the Deep Neural Net parameters.
* DQN is a form of Temporal Difference algorithm. This means that at each time step we are
trying to minimize the difference between the **target value** and the **predicted values**.
Because target values are also a function of the Deep Neural Net, in order to avoid to
chase a moving targe, we need to use a **fixed target** for updating the algorithm.
The solution is to use the same parameters of the Deep Neural Net for a few update steps.

![DQN-equation][image1]

### Hyperparameters
* Epsilon-greedy policy: at training time it's important to find the right balance
between exploration and exploitation. </br>
   * **Exploration** is intended as the ability of the agent to try different actions in
the same states and potentially discover a better policy than the current one. </br>
   * **Exploitation** is intended as the need for the agent to maximize the return, therefore
choosing the actions that lead to the maximum reward based on the current estimated policy.

   Instead of letting the agent choosing the action with the maximum expected return,
   we let it choose a random action with probability **epsilon**.
   At first epsilon is initialized to `1.0`, with a decay factor of `0.995` and capped
   to a minimum of `0.01`

* Discout Factor Gamma: Because future rewards are less predictable, it makes sense
for the agent to value immediate rewards more than future ones.
The choice for the **Discount Factor** gamma is of `0.99`

* Other hyperparameters:
    * `BUFFER_SIZE = int(1e5)  # replay buffer size`
    * `BATCH_SIZE = 64         # minibatch size`
    * `TAU = 1e-3              # for soft update of target parameters`
    * `LR = 5e-4               # learning rate`
    * `UPDATE_EVERY = 4        # how often to update the network`

### Neural Network Architecture
The architecture of the neural network is quite simple:
* 2 Hidden layers with 64 neuron each and `ReLU` activation function
* 1 output linear layer with 4 neurons, one for each possible action

### Plot of Rewards
The following image shows the average rewards for 100 episodes during training:
![Plot][image2]


### Ideas for Future Work

* *Hyperparameters tuning* Training each agent takes between 15 to 25 minutes on my
machine with an Nvidia 1060.
Because the hyperparameter space is not big, it would be interesting to use
*Bayesian Optimizers* to find the best combination of hyperparameters that would lead
to a high average score in the least amount of time.

* Another improvement would be to try to use *Double DQN*. In noisy environment
the *TD-Target* is usually overestimated. In order to avoid it, a different policy
is used for value estimation, and another one to select the next action.
This can be implemented by training 2 different value functions or by using previous
values of the parameters of the Deep Neural Net used by the agent.

* There are 2 extra algorithms that could be interested to try:
    * Prioritized experience replay
    * Dueling DQN

* *Avoid sudden turns*: After the agent has learned to collect as many yellow bananas
as possible, I noticed that the movements can be quite jerky.
I would like to add a penalty factor to the reward that is proportional to the
angle of the turn (left or right). My hope is that the agent would learn to navigate
smoothly in the environment.
