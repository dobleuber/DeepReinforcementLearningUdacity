[//]: # (Image References)

[image1]: ./Reacher.jpg "Trained Agent"

# Project 2: Continuous Control

### Introduction

In this environment, a double-jointed arm can move to target locations.
A reward of +0.1 is provided for each step that the agent's hand is in
the goal location. Thus, the goal of your agent is to maintain its
position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position,
rotation, velocity, and angular velocities of the arm. Each action is a
vector with four numbers, corresponding to torque applicable to two
joints. Every entry in the action vector should be a number between -1
and 1.

For this project, I will train an agent to control 20 identical agents.  

![Trained Agent][image1]


The task is episodic, and in order to solve the environment, your agent
must get an average score of +30 over 100 consecutive episodes.


## Setting Up the environment

### Getting Started

1. Create and activate a new environment with Python 3.6.
   * Linux or Mac:
   ```shell
   conda create --name drlnd python=3.6
   source activate drlnd
   ```
   * Windows:
   ```shell
   conda create --name drlnd python=3.6
   activate drlnd
   ```
2. You will need to install OpenAI Gym in this environment:
   ```shell
   git clone https://github.com/openai/gym.git
   cd gym
   pip install -e .
   ```

3. Clone the course repository, go to the `python/` folder and install its dependencies.
   ```shell
   git clone https://github.com/udacity/deep-reinforcement-learning.git
   cd deep-reinforcement-learning/python
   pip install .
   ```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

5. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder,
   and unzip (or decompress) the file, and change its name to Reacher-multi.app

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Crawl

After you have successfully completed the project, you might like to solve a more
difficult continuous control environment, where the goal is to teach a creature
with four legs to walk forward without falling.

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).

To solve this harder task, you'll need to download a new Unity environment.
You need only select the environment that matches your operating system:

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Continuous_Control.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
