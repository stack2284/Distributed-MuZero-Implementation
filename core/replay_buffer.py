import numpy as np 
import torch 
import ray 

@ray.remote 

class ReplayBuffer: 
    def __init__(self, config):
        self.window_size = config.window_size 
        self.batch_size = config.batch_size
        self.buffer = []
        self.games_added = 0 

    def save_game(self, game): 
        if len(self.buffer) >= self.window_size: 
            self.buffer.pop(0)
        self.buffer.append(game)
        self.games_added  += 1 
    
    def sample_batch(self, num_unroll_steps, td_steps): 
        # if len(self.buffer) < self.batch_size: 
            # return None
        
        games = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size)]
        
        batch = []
        
        for g in games: 
            game_len = len(g.history["actions"])
            
            if game_len <= num_unroll_steps: 
                index = 0
            else: 
                index = np.random.randint(0, game_len - num_unroll_steps)
            
            obs = g.history["observations"][index]
            
            actions_list = g.history["actions"][index : index + num_unroll_steps]
            
            while len(actions_list) < num_unroll_steps:
                actions_list.append(np.random.choice(g.legal_actions()))

            targets = g.make_target(index, num_unroll_steps, td_steps, 0.997)

            batch.append((obs, actions_list, targets))

        return batch
    
    def get_length(self) : 
        return len(self.buffer) 