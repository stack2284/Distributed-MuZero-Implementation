import torch
import time
import os
from config.cartpole import CartPoleConfig
from core.networks import MuZeroNetwork
from core.game import Game
from core.mcts import MCTS

def test(config, model_path=None):
    # 1. Load Config & Model
    model = MuZeroNetwork(config.observation_shape, config.action_space_size)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print("Warning: No model found. Using random weights.")

    model.eval() # Set to evaluation mode

    # 2. Setup Game with Rendering
    game = Game(env_name="CartPole-v1", render=True) 
    mcts = MCTS(config)

    print("Starting Test Game...")
    
    while not game.terminal():

        time.sleep(0.05)
        
        root = mcts.run(model, game.get_observation())
        
        action = select_best_action(root)
        
        game.apply(action)
        game.store_search_statistics(root)
        
        print(f"Step {len(game.history['actions'])} | Value: {root.value():.2f}")

    print(f"Game Over. Total Reward: {len(game.history['rewards'])}")
    game.close()

def select_best_action(node):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    _, action = max(visit_counts)
    return action

if __name__ == "__main__":
    config = CartPoleConfig()
    test(config, model_path="muzero_model.pt")