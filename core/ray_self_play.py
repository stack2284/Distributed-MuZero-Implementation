import ray 
import torch 
import numpy as np 

from core.networks import MuZeroNetwork
from core.game import Game 
from core.mcts import MCTS , Node , MinMaxStats 

@ray.remote 

class RaySelfPlayWorker : 
    def __init__(self, config , seed): 
        self.config = config 
        self.seed = seed 
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.model = MuZeroNetwork(config.observation_shape, config.action_space_size)
        self.model.eval() 
        
    def continuous_self_play(self , storage ) : 
        while True : 
            weights = ray.get(storage.get_weights.remote())
            if weights:
                self.model.load_state_dict(weights)
            game = self.play_game()
            storage.save_game.remote(game)
    
    def play_game(self):
        game = Game(self.config.env_name)
        game.reset()
        mcts = MCTS(self.config)
        while not game.terminal():
            # we assume  takes (network, observation)
            # if MCTS is hardcoded adapt it here without changing the file
            root = Node(0)
            obs = game.get_observation()
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs).float().unsqueeze(0)
            
            root.observation = obs
            min_max_stats = MinMaxStats()
            mcts.run(root, self.model, min_max_stats)
            action = self.select_action(root)
            game.apply(action)
            game.store_search_statistics(root)
            
        return game
    
    def select_action(self, node):
        # Greedy selection (simplest for now)
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        if not visit_counts:
            return 0
        _, action = max(visit_counts)
        return action
        