# Realtime Chess Game Environment for Reinforcement Learning Training

This project is a modified version of the classic chess game framework on PettingZoo, designed for use in training reinforcement learning models. Specifically, it is an infrastructure setup for an idea proposed by Dr. Tillquist at Chico State, which involves evolving decentralized real-time chess strategies.

Chess env drives are `realtime_chess_ai_dev.py` or `realtime_chess_ai.py`.  
Change moving rate/board/cooldown time in `consts.py`.

Rllib's training setup in `rllib_realtime_chess_ai.py` and testing setup in `render_rllib_realtime_chess_ai.py`.

to run the backend 
cd into chessBackend
then in the cli run python manage.py runserver


password for the admin for the django backend is 
username: admin 
password: admin  