# Decentralized Chess RL Project

This project is an RL (Reinforcement Learning) application that focuses on creating a modified version of the classic chess environment provided by PettingZoo. The modified version of the environment is inspired by Kungfu chess and introduces decentralized gameplay dynamics. The environment is then trained using the Ray RLlib library.

## Project Overview

The project takes the well-known classic chess environment from PettingZoo and adds modifications inspired by Kungfu chess. In this decentralized version of chess, the rules are adjusted to provide unique gameplay dynamics, encouraging more strategic and dynamic decision-making.

## Features

* Modified Classic Chess: The project takes the classic chess environment from PettingZoo and applies custom modifications inspired by Kungfu chess.
* Decentralized Gameplay: The chess gameplay is decentralized as each piece is its own agent, introducing new rules and dynamics to create a distinct experience.
* Ray RLlib Integration: The decentralized chess environment is trained using the Ray RLlib library for reinforcement learning.
* Training and Evaluation: The project provides facilities for training and evaluating RL agents in the modified environment.

## Getting Started
1. Clone the Repository: Clone this repository to your local machine.
2. Setup Environment: Set up the required environment for the project, including installing dependencies and creating a virtual environment.

Create a venv:  
```
python -m venv path/to/venv
# bash/zsh shell
source path/to/venv/bin/activate
# Bourne shell
. ./path/to/venv/bin/activate
```

Installing dependencies:  
```
pip install -r requirements.txt
```

3. Train with RLlib by running:
```
python rllib_realtime_chess_ai.py
```

4. Test the trained model by running:
```
python render_rllib_realtime_chess_ai.py 
``` 

## Acknowledgments

This project is built upon the foundational work of PettingZoo and Ray RLlib, which provide the classic chess environment and the reinforcement learning framework, respectively.

## Additional Notes

The idea is proposed by Dr. Tillquist at Chico State, which involves evolving decentralized real-time chess strategies.  
This project is intended for research and educational purposes.  
The decentralized chess modifications are inspired by Kungfu chess but are designed to create a unique gameplay experience.  
Ray RLlib offers powerful tools for training and evaluating RL agents in various environments.  
Chess env's drives is `realtime_chess_ai_dev.py`.  
Change moving rate/board/cooldown time in `consts.py`.