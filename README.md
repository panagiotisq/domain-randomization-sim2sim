
Domain Randomization for Sim-to-Sim Transfer

![Preview](space2.jpg)



Execution Guide


1. Requirements

Before running the project, install the required Python libraries with:
pip install -r requirements.txt


2. Files Description

main.py – Executes the agent training process.
env.py – Defines the environment, sensors, and visualization.
model.py – Defines the Actor-Critic neural network architecture.
report.pdf – Detailed project report.
requirements.txt – Contains the necessary dependencies.
README.txt – Basic project information.


3. How to Run

To start training, run:
python main.py

During training, the console displays the reward progress per epoch.
In the last 10% of the training, 3D visualization is enabled automatically.

4. Output

During training:
The best policy model (state_dict) is saved automatically.
Reward graphs are generated.

After training completes, the agent is tested on new randomized environments (zero-shot evaluation).
