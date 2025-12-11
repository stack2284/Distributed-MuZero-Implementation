import torch

class LunarLanderConfig:
    def __init__(self):
        # 1. Environment
        self.env_name = "LunarLander-v3"
        self.seed = 42
        self.action_space_size = 4
        self.observation_shape = (8,)

        # 2. Self-Play
        self.num_workers = 1
        self.num_simulations = 50  
        self.max_moves = 300
        self.discount = 0.99    

        # Exploration Noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # 3. Training
        self.batch_size = 4096
        self.learning_rate = 0.01
        self.training_steps = 5000 
        self.optimizer = "Adam"
        self.weight_decay = 1e-4

        # 4. Replay Buffer
        self.window_size = 5000
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def visit_softmax_temperature_fn(self, training_steps):
        # Explore longer for LunarLander
        if training_steps < 1000:
            return 1.0
        elif training_steps < 2000:
            return 0.5
        else:
            return 0.25