# The CartPole Environment and establishing control over the cartpole in it

Details on the environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/

**Problem Description**: 
This environment corresponds to the version of the cartpole problem described in [Neuronlike adaptive elements that can solve difficult learning control problems, Andrew Barto, Richard Sutton, CW. Anderson, 1983](https://github.com/dimitarpg13/gymnasium-demo/blob/main/docs/Neuronlike_adaptive_elements_that_can_solve_difficult_learning_control_problems_Barto1983.pdf).
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pole mechanically acts as a pendulum. The pendulum is placed upright on the cart and the goal is to balance it by applying forces in the left and right direction on the cart. The cart is free to move within the bounds of one-dimensional track. The pole is free to move only in the vertical plane of the cart and track. The controller can apply an impulsive left or right force $F$ of fixed magnitude to the cart at discrete time intervals. The cart-pole model has 4 state variables:

$x$ - position of the cart on the track.\
$\theta$ - angle of the pole with the vertical.\
$\dot{x}$ - cart velocity.\
$\dot{\theta}$ - rate of change of the angle.

The parameters of the problem are the pole length and mass, cart mass, coefficients of friction between the cart and the track, and at the hinge between the pole and the track, the impulsive control force magnitude, the force due to gravity, and the simulation step size. 
We assume that the equations of motion of the cart-pole system are not known and there is no pre-existing controller which can be immitated. 
At each time step, the controller receives a vector giving the cart-pole system's state $s \in \mathcal{S}$ at that instant. 

<img src="images/cartpole_example_Sutton_barto.png" width="300">\
Figure 1: Cart-pole example

## Random cartpole games

python code: [random_cartpole_games.py](https://github.com/dimitarpg13/gymnasium-demo/blob/main/gymnasium_demo/random_cartpole_games.py)


## Devising an algorithm to control the cartpole in the Cartpole environment


### The algorithms "Search-and-critic" and "Demon-in-box" discussed in the 1983 Barto and Sutton's article

The article delineates the presence of two _adaptive_ elements as essential piece of their algorithm: associative search element (_ASE_) and adaptive critic element (_ACE_). The presence of _ACE_ creates a reinforcement learning feedback loop which improves the performance of the algorithm. The _ASE_ must discover what responses lead to improvements in performance. The _ASE_ employs _trial-and-error_, or _generate-and-test_ search process. In the presence of input signals it generates actions by a random process. Based on feedback that evaluates the problem-solving consequences of the actions, the _ASE_ "tunes in" input signals to bias the action-generation process, conditionally on the input, so that it will more likely generate the actions leading to improved performance. The optimal action depends on the value of the input signal which is the state $s$. Actions that lead to performance improvement in the presence of specific input signals are stored in an _association map_ which is a special data structure. This stochastic search process is denoted as _associative search_ by Barto and Sutton.

The reinforcement learning process generates actions as responses to a random process that is biased by the combination of its weighted values and input patterns. To supply a signed error signal the environment must know what the actual action was and what it should have been (aka _target_).

<img src="images/reinforcement_learning_loop_Sutton_barto.png" width="600">\
Figure 2: Reinforcement learning loop

Barto and Sutton then discuss the partitioning of the original problem into a set of independent subproblems which they denote as the _"box system"_. They use identical _generate-and-test_ rule to solve each subproblem. They divide the four dimensional cart-pole state space into disjoint regions (aka _boxes_) by quantizing the four state variables. Barto and Sutton introduce up to six quantization thresholds for each state variable:

$x: \pm 0.8, \pm 2.4\ m$\
$\theta: 0, \pm 1, \pm 6, \pm 12\ \degree$\
$\dot{x}: \pm 0.5, \pm \infty\ m/s$\
$\dot{\theta}: \pm 50, \pm \infty\ \degree/s$.

This yields $3 \times 3 \times 6 \times 3 = 162$ regions corresponding to all of the combinations of the intervals. 

#### The original "Demon-in-a-Box" Algorithm for balancing the cartpole before Barto and Sutton
Each box is imagined to contain a _local demon_ whose job is to choose control action (_left_ or _right_) whenever the system state enters its box. The local demon must learn to choose the action that will tend to be correlated with long system lifeline, that is, a long time until the occurence of the failure signal. 

A _global demon_ inspects the incoming state vector at each time step and alerts the local demon whose box contains that system state. When a failure signal is received, the global demon distributes it to all local demons. Each local demon maintains estimates of the expected lifetimes of the system following a _left_ decision and following a _right_ decision. A local demon's estimate of the expected lifetime for _left_ is a weighted average of actual system lifetimes over past occasions that the system state entered the demon's box and the decision _left_ was made. The expected lifetime for the decision _right_ is determined in the same way for occasions in which a _right_ decision was made. More specifically, upon being signaled by the global demon that the system state has entered its box, a local demon does the following:

1) it chooses the control action _left_ or _right_ according to which has the longest lifetime estimate. The control system emits the control action as soon as the decision was made
2) it remembers which action was just taken and begins to count time steps
3) when a failure signal is received, it uses its current count to update the _left_ or _right_ lifetime estimate, depending on which action it chose when its box was entered.

Notice that since the effect of a demon's decision will depend on the decisions made by other demons whose boxes are visited during a trial (a trial is the time period from reset to failure), the environment of a local demon, consisting of the other demons as well as the cart-pole system, does not consistently evaluate the demon's actions.

#### Barto and Sutton's "Search-and-critic" algorithm

1. The Adaptive Search ELement (_ASE_)

Barto and Sutton chose neuron-like implementation for the _ASE_ element in their algorithm.

The local demon corresponds to the mechanism of a single neuron synapse and the output pathway of the postsynaptic element (the _ASE_) provides the common pathway for control signals. To accomplish the global demon's job of activating Barto and Sutton introduced a decoder that has four real-valued input pathways (for the system state vector) and $162$ binary valued pathways corresponding to the boxes in the original "Demon-in-a-box" algorithm. Te decoder effectively selects the synapose corresponding to the appropriate box through the $162$-components binary vector output.
 
_ASE_'s input is determined from the current cart-pole state vector by decoder that produces output vector consisting of zeros with single one indicating which one of the 162 boxes contains the state vector. _ASE_'s output determines force applied to cart. Reinforcement is constant throughout trial and becomes $-1$ to signal failure.

The other job of the global demon is to distribute a failure signal to all of the local demons - this is implemented via the reinforcement pathway of the _ASE_ element which receives the failure signal and distributes the information to all of its relevant synapses.

In more detail, the _ASE_ is defined as follows. The element has a reinforcement input pathway, $n$ pathways for nonreinforcement input, and a single output pathway (see Figure 3 below). Let $x_{i}\left(t\right), 1 \leq i \leq n$, denote the real-valued signal on the $i$-th non-reinforcement input pathway at time $t$, and let $y\left(t\right)$ denote the output at time $t$. Associated with each nonreinforcement input pathway $i$ is a real-valued weight with value at time $t$ denoted by $w_{i}\left(t\right)$.

The element's output $y\left(t\right)$ is determined from the input vector $X\left(t\right) = \left(x_{1}\left(t\right),...,x_{n}\left(t\right)\right)$ as follows:

$$y\left(t\right) = f\left[\sum_{i=1}^{n} w_{i}\left(t\right)x_{i}\left(t\right) + noise\left(t\right) \right]\quad (1)$$

Here $noise\left(t\right)$ is a real random variable with probability density function $d$ and $f$ is either a threshold, sigmoid, or identity function. or the cart-pole example, $d$ is the mean zero Gaussian distribution with variance ${\sigma}^2$, and $f$ is the following threshold function:

  $$f\left(x\right) =
    \begin{cases}
      -1 & \text{if $x \geq 0$ (control action right)}\\
      +1 & \text{if $x < 0$ (control action left)}
    \end{cases}$$  

<img src="images/ASE_element.png" width="900">\
Figure 3: The _ASE_ controller for the cart-pole system. 

According to $\left(1\right)$, actions are emitted even in the absence of nonzero input signals. The element's output is determined by chance, with a probability biased by the weighted sum of the input signals. If that sum is zero, the left and right control actions are equally probable. Assuming the decoder input shown on Figure 3, a positive weight $w_i$, for example, would make the decision right more probable than left when box $i$ is entered by the system state vector. The value of a weight, therefore, plays a role corresponding to the difference between the expected lifetimes for the left and right actions stored by a local demon in the boxes system. However, unlike the original deterministic "demon-in-a-box" algorithm, the Barto and Sutton's model is stochastic and the weight only determines the probability of an action rather than the action itself.
The learning process updates the action probabilities. The learning process updates the action probabilities. Also note that an input vector need not be of te restricted form produced by the decoder in order for $(1)$ and the equations that follow to be meaningful.
The weights $w_i, 1 \leq i \leq n$, change over discrete time as follows:

$$w_{i}\left(t+1\right) = w_{i}\left(i\right) + {\alpha}r\left(t\right){\times}e_{i}\left(t\right)    \quad (2)$$

where :

$\alpha$ is a positive constant determining the rate of change of $w_i$,

$r\left(t\right)$ is a real-valued _reinforcement_ at time $t$, and

$e_i\left(t\right)$ is _eligibility_ at time $t$ of the input pathway $i$.

The basic didea expressed by $(2)$ is that whenever certain conditions (to be discussed later) hold for the input pathway $i$, then the pathway becomes eligible to have its weight modified, and it remains eligible for some period of time after the conditions cease to hold. How $w_{i}$ changes depends on the reinforcement received during periods of eligibility. If the reinforcement indicates improved performance, then the weights of the eligible pathways are changed so as to make the element more likely to do whatever it did that made those pathways eligible. If reinforcement indicates decreased performance, then the weights of the eligible pathways are changed to make the element more likely to do something else.

_Reinforcement_: Positive $r$ indicates the occurrence of a rewarding event and a negative $r$ indicates the occurrence of a penalizing event. It can be regarded as a measure of the change in the value of a performance cirterion as commonly used in control theory. For the cartpole problem, $r$ remains zero throughout a trial and becomes $-1$ when failure occurs.

_Eligibility_: a pathway shall reach maximum eligibility a short time after the occurence of a pairing of a nonzero input signal on that pathway with the "firing" of the element. Eligibility should decay thereafter toward zero. Thus, when the consequences of the element's firing are fed back to the element, credit or blame can be assigned to the weights that will alter the firing probability when a similar input pattern occurs in the future. More generally, the eligibility of a pathway reflects the extent to which input activity on that pathway was paired in the past with element output activity. The eligibility of pathway $i$ at time $t$ is therefore a _trace_ of the product $y\left(\tau\right){\times}{x_{i}}\left(\tau\right)$ for times $\tau$ preceding $t$. If either or both of the quantities $y\left(\tau\right)$ and $x_{i}\left(\tau\right)$ are negative credit is assigned via (2) assuming the eligibility is a trace of the signed product $y\left(\tau\right){\times}{x_{i}}\left(\tau\right)$.

For computational simplicity, we generate exponentially decaying eligibility traces $e_{i}$ using the following linear difference equation:

$$e_{i}\left(t+1\right) = {\delta}{e_i}\left(t\right) + \left(1-{\delta}\right)y\left(t\right){x_i}\left(t\right)    \quad (3)$$

where $\delta, 0 \leq \delta 1$, determines the trace decay rate. Note that each synapse has its own local eligibility trace. 
   Eligibility plays a role analogous to the part of the boxes local-demon algorithm, when the demon's box is entered and an action has been chosen, remembers what action was chosen and begins to count. The factor $x_i\left(t\right)$ in $\left(3\right)$ triggers the eligibility trace, a kind of count, or contributes to an ongoing trace, whenever box $i$ is entered ($x_i\left(t\right)=1$).
   Unlike the count initiated by a local demon in the "demon-in-a-box" algorithm, the eligibility trace effectively counts down rather than up (more precisely, its magnitude decays toward zero). Recall that reinforcement $r$ remains zero until a failure occurs, at which time it becomes $-1$. Thus whatever control decision was made when a box was visited will always be made _less_ likely when the failure occurs, but the longer the time interval between the decision and the ocurrence of the failure signal, the less this decrease in probability will be. Since the failure signal always eventually occurs, the action that was taken may deserve some of the blame for the failure. Despite the fact that all actions inevitable lead to failure, one action is _probably_ better than the rest. The learning process defined by $\left(1\right)-\left(3\right)$ needs to be more subtle to ensure convergence to the actions that yeld the least penality in cases in which only pennalization is avaialable. This subtelty is implemented via _ACE_ rather than in _ASE_ as it will be clarified in the section on _ACE_.
   
2. The Adaptive Critic Element (_ACE_)

The Fig. 4 shows an _ASE_ coupled with an _ACE_ for the cartpole task. The _ACE_ receives the externally supplied reinforcement signal which it uses to determine how to compute, on th ebasis of the current cartpole state vector, an improved reinforcement signal (denoted with $\hat{r}$ on the Figure) that it sends to the _ASE_.
Among its other functions, the _ACE_ constructs predictions of reinforcement so that if penalty is less than its expected level, it acts as a reward. As implied earlier _ASE_ operates in conjunction witht the _ACE_. The _ACE_ stores in each box a prediction or expectation of the reinforcement that can eventually be obtained from the environment by choosing an action for that box. The _ACE_ uses this prediction to determine a reinforcement signal that it delivers to the _ASE_ whenever the box is entered by the cartpole state, thus permitting learning to occur throghout the trials rather than solely on failure. This greatly decreases the uncertainty faced by the _ASE_. The cnetral idea behind the _ACE_ algorithm is that predictions are formed that predict not just reinforcement but also future predictions of reinforcement (_TODO: elaborate what that means_).

Like the _ASE_, the _ACE_ has a reinforcement input pathway, _n_ pathways for nonreinforcement input, and a single output pathway (Figure 4). 
Let $r\left(t\right)$ denote the real-valued reinforcement at time $t$. Let $x_i\left(t\right)$, $1 \leq i \leq n$, denote the real-valued signal on the $i$-th nonreinforcement input pathway at time $t$. Let $\hat(t)\left(t\right)$ denote the real-valued output signal at time $t$. Each nonreinforcement input pathway $i$ has a weight with real value $v_i\left(t\right)$ at time $t$. The output $\hat(t)$ is the improved reinforcement signal that is used by the _ASE_ in place of $r$ in (2). 
In order to produce $\hat(r)$, the _ACE_ must determine a prediction $p\left(t\right)$ of evetual reinforcement that is a fucntion of the input vector $X\left(t\right)$ (which in the boxes paradygm, simply selects a box). We let :

$$p\left(t\right) = \sum_{i=1}^{n} v_{i}\left(t\right) x_{i}\left(t\right)    \quad (4)$$

and seek a means of updating the weights $v_i$ so that $p\left(t\right)$ converges to an accurate prediction. 


<img src="images/ASE_and_ACE_elements.png" width="900">\
Figure 4: The _ASE_ controller and the _ACE_ element for the cart-pole system. 

The updating rule we use is:

$$v_{i}\left(t+1\right) = v_{i}\left(t\right) + \beta\left[r\left(t\right) + \gamma p\left(t\right) - p\left(t-1\right)\right]\bar{x}_{i}\left(t\right)   \quad (5)$$

where $\beta$ is a positive constant determining the rate of change of $v_i$; $\gamma$, $0 \le \gamma \leq 1$, is 


_//TODO: finish this section_

#### Advantages of the "Search-and-critic" algorithm over the "Demon-in-a-box" algorithm
  
   Although the boxes system and the cartpole problem serves the purpose of explaining the ASE design in understandable way, the _ASE_ does not represent an attempt to duplicate the "demon-in-a-box" algorithm in neuronlike form. The _ASE_ formulation is less restricted than the "demon-in-a-box" algorithm in several ways. First, the "demon-in-a-box" algorithm is based on the subdivision of the problem space into a finire number of nonoverlapping regions, and no generalization is attempted between regions. It develops a control rule that is effectively specified by means of a lookup table. Although a form of generalization can be applied to the "Demon-in-a-box" algorithm by using an averaging process over neighboring boxes it is not immediately obvious how to extend the algorithm to take advantage of the other forms of generalization that would be possible if the controlled system's states could be represented by arbitrary vectors rather than only by the standard unit basis vectors which are produced by a suitable decoder. The _ASE_ can accept arbitrary input vectors and can be regarded as a step toward extending the type of generalization produced by error-corection supervised learning pattern classification methods to the less restricted reinforcement learning paradigm.
   The "Demon-in-box" algorithm is also restricted in that its design was based on the _a priori_ knowledge that the time until failure was to serve as he evaluation criterion and that the learning process would be divided into distinct trials that would always end with a failure signal. This _a priori_ knowledge was used to reduce the uncertainty in the problem by restricting each local demon to choosing the same action each time its box was entered during any given trial. The _ASE_, on other hand, is capable of working to achieve rewarding events and to avoid penalizing events which might occur at any time. It is not exclusively failure-driven, and its operation is specified without reference to the notion of a trial (_TODO_: _need to elaborate on the latter_).    
   
   
### Using and Implementing Deep Q Network

We use $Q$ function to define a target for the current state $s$.

$$loss = \left( r + \gamma \max_{a' \in \mathcal{A}} Q'\left(s,a'\right) - Q\left(s,a\right)\right)^{2}$$

