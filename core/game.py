import gymnasium as gym 
import numpy as np 
import torch 

class AbstractGame : 
    def __init__ (self , seed = None) : 
        self.env = None
        self.action_space_size = 0 
        self.observation = None 
        self.seed = seed 
        self.history = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "policies": [],
            "values": []
        }
        # currently hardcoded 
        self.discount = 0.99
        
    def step (self , action) : 
        raise NotImplementedError
    
    def legal_actions(self) : 
        raise NotImplementedError
    def reset(self) : 
        raise NotImplementedError
    def close(self) : 
        if self.env : 
            self.env.close()
    
    def store_search_stats (self , root , action_space_size) : 
        sum_visits = sum(child.visit_count for child in root.children.values())
        policy = [0.0] *action_space_size 
        
        if sum_visits > 0 : 
            for action , child in root.children.items() : 
                policy[action] = child.visit_count / sum_visits
        else : 
            policy = [1.0 / action_space_size] * action_space_size
        
        self.history["policies"].append(policy) 
        self.history["values"].append(root.value())
        
    def make_image(self, i):
        # Returns the observation at index i
        return self.history["observations"][i]
        
    def store_search_statistics(self, root): 
        sum_visits = sum(child.visit_count for child in root.children.values())
        policy = [0.0] * self.action_space_size 
        
        if sum_visits > 0: 
            for action, child in root.children.items(): 
                policy[action] = child.visit_count / sum_visits
        else: 
            policy = [1.0 / self.action_space_size] * self.action_space_size
        
        self.history["policies"].append(policy) 
        self.history["values"].append(root.value())
    
    def make_target (self , state_index , num_unrool_steps , td_steps , discount) :
        """
        generates target for trainging 
        in each unrool step we need 
        the value target who won 
        reward target did we get points 
        the policy target what mcts did 
        """
        # old serial training 
        # targets = [] 
        # for current_index in range(state_index , state_index + num_unrool_steps + 1) : 
        #     bootstrap_index = current_index  + td_steps 
        #     value = 0 
        #     if bootstrap_index < len(self.history["values"]) : 
        #         value = self.history["values"][bootstrap_index] *((discount)**td_steps) 
            
        #     for i , reward in enumerate(self.history["rewards"][current_index : bootstrap_index]) : 
        #         value += reward * (discount ** i) 
            
            
        #     if current_index < len(self.history["rewards"]) : 
        #         last_reward = self.history["rewards"][current_index]
        #         if current_index < len(self.history["policies"]) : 
        #             policy = self.history["policies"][current_index] 
        #         else : 
        #             policy = [1.0 / self.action_space_size]*self.action_space_size 
        #     else : 
        #         last_reward = 0 
        #         policy = [1.0 / self.action_space_size ] * self.action_space_size
        #     targets.append((value , last_reward , policy))
        
        # return targets         
        bootstrap_index = state_index + td_steps
        
        value = 0
        if bootstrap_index < len(self.history["rewards"]):
            # Discounted sum of rewards
            for i, r in enumerate(self.history["rewards"][state_index:bootstrap_index]):
                value += r * (discount ** i)
            # Add bootstrap value (simplified for now as 0, strictly should be network value)
            value += 0 
        else:
            value = 0 
            
        # 2. Target Reward
        if state_index < len(self.history["rewards"]):
            reward = self.history["rewards"][state_index]
        else:
            reward = 0

        # 3. Target Policy (The FIX is here)
        # We use history["policies"], not child_visits
        if state_index < len(self.history["policies"]):
            policy = self.history["policies"][state_index]
        else:
            policy = [1/self.action_space_size] * self.action_space_size

        return value, reward, policy

    
    

class Game(AbstractGame) : 
    def __init__(self, env_name , seed=None , render=False):
        super().__init__(seed) 
        self.done = False
        render_mode = 'human' if render else None 
        self.env = gym.make(env_name , render_mode=render_mode) 
        
        if self.seed is not None : 
            self.env.reset(seed=self.seed) 
        
        if isinstance(self.env.action_space , gym.spaces.Discrete) :
            self.action_space_type = 'discrete' 
            self.action_space_size = self.env.action_space.n        
        elif isinstance(self.env.action_space , gym.spaces.Box) : 
            self.action_space_type = 'continuous'
            self.action_space_size = self.env.action_space.shape[0]
            self.action_min = self.env.action_space.low 
            self.action_max = self.env.action_space.high 
        else : 
            raise ValueError("Action space error type") 
        
        if len(self.env.observation_space.shape) == 3 : 
            h , w , c = self.env.observation_space.shape 
            self.observation_shape = (c ,h , w) 
            self.is_image = True 
        else : 
            self.observation_shape = self.env.observation_space.shape
            self.is_image = False 
    
    def reset(self) :
        observation , info = self.env.reset()
        processed_obs = self._process_obs(observation)
        
        self.history = {
            "observations": [processed_obs],
            "actions": [],
            "rewards": [],
            "policies": [],
            "values": []
        }

        self.history["rewards"].append(0)
        return processed_obs
    
    def step(self , action ) : 
        """
        Excetues actions 
        inp is int for discrete and is floation point for contineeous
        """
        if self.action_space_type == 'continuous' : 
            real_action = np.clip(action , self.action_min , self.action_max )
        else : 
            real_action = action 
        observation , reward , terminated , truncated , info = self.env.step(real_action)
        done = terminated or truncated 
        processed_obs = self._process_obs(observation)
        
        self.history["actions"].append(action )
        self.history["rewards"].append(reward)
        self.history["observations"].append(processed_obs)
        
        return processed_obs , reward , done 
        
        
    def legal_actions(self) : 
        if self.action_space_type == 'discrete' :  
            return list(range(self.action_space_size))
        else : 
            return None 
    def _process_obs(self , observation) :  
        if self.is_image : 
            # using float 32 for better -performance on m4 mac 
            obs = np.array(observation , dtype=np.float32).transpose(2 , 0 , 1)/255.0
        else : 
            obs = np.array(observation , dtype=np.float32) 
        return obs 
    
    def action_to_string (self , action ) : 
        if self.action_space_type == 'discrete' : 
            return str(action) 
        else : 
            return f"{['{:.2f}'.format(x) for x in action]}"
        # this for cont else will round and give error
    
    def terminal(self):
        return self.done

    def apply(self, action):
        _, _, self.done = self.step(action)

    def get_observation(self):
        if len(self.history["observations"]) < 1 :
            return None
        return self.history["observations"][-1]
           
