import ray
import time
import sys
import os

print("--> Script Starting...", flush=True)

try:
    from config.cartpole import CartPoleConfig 
    from core.networks import MuZeroNetwork
    from core.ray_storage import SharedStorage
    from core.ray_self_play import RaySelfPlayWorker
    from core.ray_trainer import RayTrainer
    print("--> Imports Successful", flush=True)
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import files. {e}", flush=True)
    sys.exit(1)

if __name__ == "__main__":
    print("--> Initializing Ray...", flush=True)
    # Added _temp_dir='/tmp/ray' which helps on Mac M-series permissions
    ray.init(ignore_reinit_error=True, _temp_dir='/tmp/ray')
    print("--> Ray Initialized", flush=True)
    
    config = CartPoleConfig()
    
    print("--> Starting Storage Actor...", flush=True)
    storage = SharedStorage.remote(config)
    
    # --- FIX: Pass specific arguments instead of just 'config' ---
    print("--> Setting Initial Weights...", flush=True)
    temp_model = MuZeroNetwork(config.observation_shape, config.action_space_size)
    
    # Extract weights manually since we didn't change your networks.py
    cpu_weights = {k: v.cpu() for k, v in temp_model.state_dict().items()}
    ray.get(storage.set_weights.remote(cpu_weights))
    
    print("--> Starting Trainer Actor...", flush=True)
    trainer = RayTrainer.remote(config)
    trainer.continuous_training.remote(storage)
    
    print("--> Starting Worker Actors...", flush=True)
    num_workers = 2 
    workers = [RaySelfPlayWorker.remote(config, i) for i in range(num_workers)]
    
    for w in workers:
        w.continuous_self_play.remote(storage)
        
    print(f"--> System Running with {num_workers} workers.", flush=True)
    
    while True:
        try:
            info = ray.get(storage.get_info.remote(), timeout=10)
            print(f"Storage Info: {info}", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"Error in monitor loop: {e}", flush=True)
            break