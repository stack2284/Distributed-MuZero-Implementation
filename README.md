# Distributed MuZero (Ray + PyTorch)

A modular, distributed implementation of the MuZero reinforcement learning algorithm.  
This project uses Ray to parallelize data generation (self-play) and training, and PyTorch for the neural networks.

It is currently configured for the CartPole-v1 environment but is designed for extension to other Gymnasium environments.

---

## Features

* Distributed architecture: self-play workers, shared storage, and a trainer.
* Asynchronous training: data generation and gradient updates occur concurrently.
* Full Monte Carlo Tree Search (MCTS) implementation using learned models.
* Hardware acceleration with automatic device selection (MPS, CUDA, or CPU).
* Visual evaluation script for rendering the trained agent.

---

## Project Structure

```
MuZero/
├── config/
│   └── cartpole.py         # Configuration settings (hyperparameters)
├── core/
│   ├── game.py             # Gymnasium wrapper and episode history
│   ├── mcts.py             # Monte Carlo Tree Search implementation
│   ├── networks.py         # PyTorch networks: representation, dynamics, prediction
│   ├── ray_self_play.py    # Self-play worker actor
│   ├── ray_storage.py      # Central storage: replay buffer and model weights
│   └── ray_trainer.py      # Trainer actor: loss computation and optimization
├── main_distributed.py     # Entry point for distributed training
└── test_agent.py           # Script to load and visually evaluate the trained agent
```

---

## Installation

1. Clone or create the repository.
2. Install the dependencies:

```
pip install torch gymnasium numpy "ray[default]"
```

---

## Usage

### 1. Train the Agent

This command launches a local Ray cluster and starts the storage, trainer, and multiple self-play workers.

```
python main_distributed.py
```

Notes:

* Storage statistics and training loss are printed periodically.
* Ray dashboard is typically available at: http://localhost:8265 
* can be configured to act at a seprate port if using multiple devices 
* The model is saved to `muzero_model.pt` every 50 training steps.

---

### 2. Watch the Agent Play

Once `muzero_model.pt` has been generated, run:

```
python test_agent.py
```

This opens a visual environment window and displays the learned behavior.

---

## Configuration

Modify `config/cartpole.py` to change hyperparameters.

Important parameters include:

* `num_workers` (in `main_distributed.py`): number of self-play actors
* `num_unroll_steps`: how many steps of future prediction the model is trained on
* `td_steps`: temporal difference horizon
* `window_size`: replay buffer capacity (FIFO)

