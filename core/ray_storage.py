import ray 
import numpy  as np 

@ray.remote 

class SharedStorage : 
    def __init__(self, config ):
        self.config = config 
        self.buffer = {} 
        self.weights = None 
        self.game_count = 0 
        
    def get_weights(self) : 
        return self.weights 
    
    def set_weights(self , weights ) : 
        self.weights = weights 
    
    def save_game(self , game) :
        self.buffer[self.game_count] = game 
        self.game_count += 1 
        while len(self.buffer) > self.config.window_size:
            try:
                oldest_key = min(self.buffer.keys())
                del self.buffer[oldest_key]
            except ValueError:
                break
        
        return self.game_count
        
    
    def sample_batch(self):
        """Trainer calls this to get a batch of data."""
        games_list = list(self.buffer.values())
        
        # Wait until we have enough data
        if len(games_list) < self.config.batch_size:
            return None

        batch = []
        for _ in range(self.config.batch_size):
            # 1. Pick a random game
            game = games_list[np.random.choice(len(games_list))]
            
            # 2. Pick a random position in that game
            # We need to make sure we don't pick the very last step, 
            # so we have room to unroll.
            game_len = len(game.history)
            # Default to 0 if game is too short (safety check)
            if game_len > self.config.num_unroll_steps:
                game_pos = np.random.choice(game_len - self.config.num_unroll_steps)
            else:
                game_pos = 0
            # Append TUPLE (Game, index)
            batch.append((game, game_pos))
            
        return batch
    
    
    def get_info (self) : 
        return {
            "total_games" :self.game_count , 
            "buffer_size" : len(self.buffer) 
        }
        
        
        