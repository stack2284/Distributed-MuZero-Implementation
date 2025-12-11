import torch 
import torch.nn.functional as F 
import numpy as np 


class MuZeroTrainer : 
    def __init__(self , config , network , replay_buffer):
        self.config = config
        self.network = network
        self.replay_buffer  = replay_buffer 
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay 
        )
        
    def train_step(self) : 
        batch = self.replay_buffer.sample_batch(num_unroll_steps=5 , td_steps=10)
        # handle small buffer execption 
        if not batch : return 0 
        
        observation = torch.tensor(np.array([b[0] for b in batch ]), dtype=torch.float32 , device=self.config.device)
        actions_list = [b[1] for b in batch] 
        target_list = [b[2] for b in batch]         
        
        hidden_state =self.network.representation(observation) 
        policy_logits , value = self.network.prediction(hidden_state)
        total_loss = 0  
        total_loss += self._calculate_step_loss(0 , policy_logits , value , None , target_list)
        
        
        
        for k in range(5)  : 
            actions_k = torch.tensor(
                [[game_actions[k]] for game_actions in actions_list ],
                dtype=torch.float32 ,
                device=self.config.device 
                )

            hidden_state , reward_pred = self.network.dynamics(hidden_state, actions_k)
            
            policy_logits , value = self.network.prediction(hidden_state)
            
            hidden_state.register_hook(lambda grad : grad * 0.5 )
            total_loss += self._calculate_step_loss(k +  1 , policy_logits , value , reward_pred , target_list)
        
        self.optimizer.zero_grad()         
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
        self.optimizer.step()
        return total_loss.item()
    
    def _calculate_step_loss(self , step_index , policy_logits , value_pred , reward_pred , targets_list) : 
        target_values = torch.tensor([t[step_index][0] for t in targets_list], dtype=torch.float32, device=self.config.device).unsqueeze(1)
        target_rewards = torch.tensor([t[step_index][1] for t in targets_list], dtype=torch.float32, device=self.config.device).unsqueeze(1)
        target_policies = torch.tensor(np.array([t[step_index][2] for t in targets_list]), dtype=torch.float32, device=self.config.device)
         
        value_loss = F.smooth_l1_loss(value_pred, target_values)
        
        if reward_pred is not None : 
            reward_loss  = F.smooth_l1_loss(value_pred, target_values)
        else : 
            reward_loss = 0 
        
        policy_loss = -(target_policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        
        return policy_loss + reward_loss + value_loss
         
         