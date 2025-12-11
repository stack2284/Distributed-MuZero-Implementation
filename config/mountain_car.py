import torch

class AcrobotConfig:
    def __init__(self):
        # 1. Environment
        self.env_name = "Acrobot-v1"
        self.seed = 42
        self.action_space_size = 3
        self.observation_shape = (6,)

        # 2. Self-Play
        self.num_workers = 1
        self.num_simulations = 50
        self.max_moves = 500
        self.discount = 0.997

        # Exploration Noise
        self.root_dirichlet_alpha = 0.3 
        self.root_exploration_fraction = 0.25

        # 3. Training
        self.batch_size = 64
        self.learning_rate = 0.002
        self.training_steps = 3000
        self.optimizer = "Adam"
        self.weight_decay = 1e-4

        # 4. Replay Buffer
        self.window_size = 2000
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def visit_softmax_temperature_fn(self, training_steps):
        if training_steps < 500:
            return 1.0
        elif training_steps < 1000:
            return 0.5
        else:
            return 0.25