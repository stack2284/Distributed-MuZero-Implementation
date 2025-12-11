import torch

class AcrobotConfig:
    def __init__(self):
        # 1. Environment
        self.env_name = "Acrobot-v1"
        self.seed = 42
        self.action_space_size = 3      # Actions: Torque -1, 0, +1
        self.observation_shape = (6,)   # Cos/Sin of angles, velocities

        # 2. Self-Play
        self.num_workers = 1
        self.num_simulations = 50       # 50 is good for Acrobot
        self.max_moves = 500            # Game truncates at 500 steps
        self.discount = 0.997           # High discount to value the distant goal

        # Exploration Noise
        self.root_dirichlet_alpha = 0.3 # Slightly higher noise for 3 actions
        self.root_exploration_fraction = 0.25

        # 3. Training
        self.batch_size = 256
        self.learning_rate = 0.002      # Slightly lower LR usually helps Acrobot
        self.training_steps = 2000      
        self.optimizer = "Adam"
        self.weight_decay = 1e-4

        # 4. Replay Buffer
        self.window_size = 2000         # Keep more history
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def visit_softmax_temperature_fn(self, training_steps):
        # Exploration schedule
        if training_steps < 500:
            return 1.0
        elif training_steps < 1000:
            return 0.5
        else:
            return 0.25