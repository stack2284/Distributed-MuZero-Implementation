import torch 

class CartPoleConfig : 
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.seed = 22 
        self.action_space_size = 2
        self.observation_shape = (4 , ) 
        
         # self play data gen here 
        self.num_workers = 1 
        self.num_simulations = 25
        self.max_moves = 500
        self.discount = 0.997 
        
        # exploration noice dirichlet 
        # adds randomness to the root node to encourage trying diffrent moves 
        self.root_dirichlet_alpha = 0.25 
        self.root_exploration_fraction = 0.25 
        
        self.num_unroll_steps = 5  
        self.td_steps = 10
        
        # training params 
        self.batch_size = 128
        self.buffer_size = 128
        self.learning_rate = 0.005
        self.lr = self.learning_rate  
        self.training_steps = 2000 
        self.optimizer = 'Adam' 
        self.weight_decay = 1e-4 
        
        self.window_size = 1000 
        Device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
        self.device = torch.device(Device)
    
    def visit_softmax_temperature_fn (self , trained_steps ) : 
        if(trained_steps < 500) : 
            return 1 
        elif trained_steps < 750: 
            return 0.5 
        else : 
            return 0.25 
        
        