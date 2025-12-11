import ray
import torch
import torch.nn.functional as F
import numpy as np
import time
from core.networks import MuZeroNetwork

@ray.remote
class RayTrainer:
    def __init__(self, config):
        self.config = config
        self.model = MuZeroNetwork(config.observation_shape, config.action_space_size)
        
        # --- AUTO-DETECT DEVICE ---
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"Trainer running on device: {self.device}")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.model.train()
        self.training_step = 0

    def continuous_training(self, storage):
        while True:
            batch = ray.get(storage.sample_batch.remote())
            
            if not batch:
                time.sleep(0.5)
                continue

            loss_info = self.update_weights(batch)
            self.training_step += 1
            
            if self.training_step % 50 == 0:
                cpu_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
                storage.set_weights.remote(cpu_weights)
                torch.save(cpu_weights, "muzero_model.pt")
                
                print(f"Step {self.training_step} | "
                      f"Loss: {loss_info['total_loss']:.4f} | "
                      f"Value: {loss_info['value_loss']:.4f} | "
                      f"Policy: {loss_info['policy_loss']:.4f}", flush=True)

    def update_weights(self, batch):
        self.optimizer.zero_grad()
        
        total_loss = 0
        value_loss_sum = 0
        policy_loss_sum = 0
        
        for game, start_index in batch:
            # 1. Initial Observation
            obs_data = game.make_image(start_index)
            obs = torch.tensor(obs_data, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            hidden_state = self.model.representation(obs)
            
            for k in range(self.config.num_unroll_steps):
                target_index = start_index + k
                
                # A. Predict
                policy_logits, value_pred = self.model.prediction(hidden_state)
                
                # B. Targets
                target_value, target_reward, target_policy = game.make_target(
                    target_index, self.config.num_unroll_steps, self.config.td_steps, self.config.discount
                )
                
                # --- FIX 1: Tensor creation on Device ---
                t_value = torch.tensor(target_value, dtype=torch.float32, device=self.device)
                
                # --- FIX 2: UNSQUEEZE POLICY TARGET ---
                # Changes shape from [2] to [1, 2] to match policy_logits
                t_policy = torch.tensor(target_policy, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # C. Loss
                v_loss = F.mse_loss(value_pred.squeeze(), t_value)
                p_loss = F.cross_entropy(policy_logits, t_policy)
                
                # Scale gradient
                hidden_state.register_hook(lambda grad: grad * 0.5)

                total_loss += (v_loss + p_loss)
                value_loss_sum += v_loss.item()
                policy_loss_sum += p_loss.item()

                # D. Dynamics
                if k < self.config.num_unroll_steps - 1:
                    if target_index < len(game.history["actions"]):
                        action_id = game.history["actions"][target_index] 
                        
                        action_tensor = torch.tensor([[action_id]], dtype=torch.float32, device=self.device)
                        hidden_state, reward_pred = self.model.dynamics(hidden_state, action_tensor)
                        
                        if (target_index) < len(game.history["rewards"]):
                            true_r = game.history["rewards"][target_index]
                            t_reward = torch.tensor(true_r, dtype=torch.float32, device=self.device)
                            
                            r_loss = F.mse_loss(reward_pred.squeeze(), t_reward)
                            total_loss += r_loss
                    else:
                        break

        total_loss = total_loss / len(batch)
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss_sum / len(batch),
            "policy_loss": policy_loss_sum / len(batch)
        }