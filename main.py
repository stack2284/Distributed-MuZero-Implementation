from config.cartpole import CartPoleConfig 
from core.game import Game 
from core.networks import MuZeroNetwork 
from core.mcts import MCTS , MinMaxStats , Node 
from core.replay_buffer import ReplayBuffer 
import numpy as np 
from core.trainer import MuZeroTrainer 
import torch 
from config.lunar_lander import LunarLanderConfig

def play_game (config , network , replay_buffer) : 
    game = Game(config.env_name , seed=config.seed , render=False) 
    mcts =  MCTS(config)
    observation = game.reset()
    done = False 
    total_reward = 0  
    
    # print("staring game") 

    
    # main loop 
    
    while not done : 
        # inti  root node 
        root = Node(0)
        root.observation = observation 
        
        min_max_stats = MinMaxStats() 
        with torch.no_grad():
            mcts.run(root , network , min_max_stats)
        game.store_search_stats(root, config.action_space_size)
        
        # action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        visits = [
           root.children[a].visit_count if a in root.children else 0 for a in range(config.action_space_size)]
        
        #  this parameter is for temp keep low for training on laptop or personally use func for months of training 
        # ideal keep 15 for cart pole and 
        # 5 for lunar lander and other complex games 
        temperature = 1.0 if len(game.history["actions"]) < 5 else 0.0
        # temperature = config.visit_softmax_temperature_fn(len(game.history["actions"]))
        if temperature == 0: 
            action = np.argmax(visits)
        else:
            visits = np.array(visits, dtype=float)
            probs = visits / visits.sum()
            action = np.random.choice(len(probs), p=probs)
        
        observation , reward , done = game.step(action) 
        total_reward += reward 
        
        # print(f"Action {action} Reward {reward}, Done {done}") 
    
    # print("Game Over") 
    replay_buffer.save_game(game)
    game.close()
    
    return total_reward 
    
    
if __name__ == "__main__" : 
    # config = CartPoleConfig()
    #savefile = "muzero_cartpole.pt"
    config = LunarLanderConfig()
    savefile = "muzero_lunarlander.pt"
    
    replay_buffer = ReplayBuffer(config)
    network = MuZeroNetwork(
        config.observation_shape, 
        config.action_space_size
    )
    trainer = MuZeroTrainer(config , network , replay_buffer )    
    
    print("----startimg training loop----")
    
        
    for i in range(1000) : 
        score = play_game(config , network , replay_buffer) 
        print( f"game { i + 1} | score : {score} | Buffer : {len(replay_buffer.buffer)} ")
        if len(replay_buffer.buffer) > 5 : 
            loss = 0 
            for _ in range(20) : 
                loss += trainer.train_step() 
            
            print(f"     Training loss : {loss / 20:.4f}")
    
    torch.save(network.state_dict(), savefile)
    print("Model saved to " , savefile)
    print("done")
    # To load and watch:
    # network.load_state_dict(torch.load("muzero_cartpole.pt", map_location=config.device))
    # play_game(config, network, replay_buffer, render=True) # Set render=True!