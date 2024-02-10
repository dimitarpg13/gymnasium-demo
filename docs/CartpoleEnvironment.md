# The CartPole Environment and establishing control over the cartpole in it

Details on the environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/

**Description**: 
This environment corresponds to the version of the cartpole problem described in [Neuronlike adaptive elements that can solve difficult learning control problems, Andrew Barto, Richard Sutton, CW. Anderson, 1983](https://github.com/dimitarpg13/gymnasium-demo/blob/main/docs/Neuronlike_adaptive_elements_that_can_solve_difficult_learning_control_problems_Barto1983.pdf).
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pole mechanically acts as a pendulum. The pendulum is placed upright on the cart and the goal is to balance it by applying forces in the left and right direction on the cart.


## Random cartpole games

python code: [random_cartpole_games.py](https://github.com/dimitarpg13/gymnasium-demo/blob/main/gymnasium_demo/random_cartpole_games.py)


## Devising an algorithm to control the cartpole in the Cartpole environment


### Idea from for an algorithm

We shall have two main components in our algorithm -


### Using and Implementing Deep Q Network

$$loss = \left( r + \gamma \max_{a' \in \mathcal{A}} Q'\left(s,a'\right) - Q\left(s,a\right)\right)^{2}$$

