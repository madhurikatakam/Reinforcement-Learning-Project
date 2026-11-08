# Hierarchical Reinforcement Learning for Adaptive and Resource-Aware Federated Learning

## 1. Project Overview

Federated Learning (FL) enables multiple clients to collaboratively train a global machine learning model without sharing their raw data. However, traditional Federated Averaging (FedAvg) generally uses predefined client-selection strategies and fixed local training epochs.

In real-world environments, clients have heterogeneous computational resources, network conditions, battery levels, data distributions, and availability.

This project proposes a **three-agent Reinforcement Learning framework** to adaptively optimize federated learning decisions.

The framework consists of three cooperating agents:

1. **Client Count Agent:** Determines the number of clients that should participate in each communication round.
2. **Client Selection Agent:** Selects the most suitable clients based on their data characteristics, resource availability, and participation history.
3. **Local Epoch Agent:** Determines the number of local training epochs for each selected client.

The framework aims to balance global model accuracy, computational cost, communication overhead, energy consumption, data utility, and participation fairness.

## 2. Project Objectives

- Develop an adaptive client participation strategy.
- Optimize client selection under non-IID data distributions.
- Dynamically allocate local training epochs.
- Consider heterogeneous client resources and availability.
- Balance model accuracy and resource utilization through multi-objective reward functions.
- Evaluate different RL algorithms and hybrid configurations.
- Investigate the impact of client dropout and simulated malicious updates.

## 3. Project Architecture

The proposed framework follows a sequential three-agent decision-making process.

### Agent 1: Client Count Agent
Determines the number of clients participating in a federated learning round.

### Agent 2: Client Selection Agent
Selects the required number of distinct available clients using client-level information.

### Agent 3: Local Epoch Agent
Assigns a suitable number of local training epochs to each selected client.

### Federated Learning Process

1. Initialize the global model and client resources.
2. Distribute the global model to selected clients.
3. Perform local model training.
4. Collect local model updates.
5. Apply the simulated malicious-update filtering mechanism.
6. Aggregate accepted updates using FedAvg.
7. Evaluate global model performance.
8. Calculate rewards using accuracy improvement, data utility, fairness, time, communication, and energy.
9. Update the RL agents.
10. Continue to the next communication round.

## 4. Algorithms Implemented

The project evaluates seven configurations:

| Configuration | Count Agent | Selection Agent | Epoch Agent |
|---|---|---|---|
| FedAvg | Fixed | Random | Fixed |
| QLEARNING3 | Q-Learning | Q-Learning | Q-Learning |
| DQN3 | DQN | DQN | DQN |
| DDQN3 | DDQN | DDQN | DDQN |
| PPO3 | PPO | PPO | PPO |
| DQN_PPO | DQN | DQN | PPO |
| PPO_DQN | PPO | DQN | DQN |

The PPO_DQN configuration represents the proposed hybrid approach.

## 5. Experimental Setup

| Parameter | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Number of clients | 50 |
| Communication rounds | 50 |
| Number of classes | 10 |
| Data distribution | Non-IID |
| Dirichlet concentration (α) | 0.3 |
| Local training epochs | 1–5 |
| FedAvg local epochs | 2 |
| Local batch size | 64 |
| Local optimizer | Adam |
| Learning rate | 0.001 |
| Discount factor (γ) | 0.95 |
| Client CPU range | 1–8 |
| Battery range | 20–100 |
| Bandwidth range | 5–100 Mbps |
| Latency range | 20–200 ms |

Client resources and availability are simulated and updated during training.

### Reward Function

The high-level reward considers:

- Global accuracy improvement
- Data utility
- Client participation fairness
- Training time
- Communication cost
- Energy consumption

The local reward considers individual client accuracy improvement and resource costs.

Jain's Fairness Index is used to measure participation fairness.

## 6. Additional Experiments

The ablation and robustness notebook includes the following experiments:

### Ablation Study
- Full proposed PPO_DQN framework.
- Fixed client-count configuration.
- Random client-selection configuration.
- Fixed local-epoch configuration.

### Reward Sensitivity Analysis
Different reward-weight configurations are defined to study the trade-off between accuracy, data utility, resource consumption, and fairness.

### Robustness Analysis
The framework is evaluated under different client availability and dropout conditions, including normal, moderate-dropout, and severe-dropout scenarios.

### Additional Evaluation
- Client participation distribution.
- Jain's fairness index.
- Communication volume estimation.
- Resource-group participation.
- Accuracy-fairness trade-off.
- Per-class test accuracy.

**Notebook descriptions:**

- `RL_Algorithm_Comparison_.ipynb`: Implements the federated learning environment, RL agents, seven algorithm configurations, training, and performance comparison against FedAvg.
- `Ablation_study.ipynb`: Extends the main experiments with ablation configurations, reward sensitivity, robustness testing, and additional evaluation metrics.

## 7. Dependencies

The project uses Python and the following libraries:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd Hierarchical-RL-Federated-Learning
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch and Torchvision builds for your CPU or CUDA environment if needed.

### Step 4: Open the notebooks

```bash
jupyter notebook
```

Alternatively, upload the notebooks to Google Colab.

### Step 5: Configure the dataset

The project uses CIFAR-10 through Torchvision.

The dataset can be downloaded automatically using:

```python
torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=True,
    download=True,
    transform=transform
)
```

Set `DATA_PATH` to a valid local directory or Google Drive directory before execution.

If running in Google Colab, execute the Google Drive mounting cell and configure the dataset path.

### Step 6: Execute the experiments

Run the notebooks sequentially, starting with:

1. `RL_Algorithm_Comparison_.ipynb`
2. `Ablation_study.ipynb`

Execute the initialization, dataset loading, model setup, and agent definition cells before running the experiments.

Run the short smoke test before starting the complete experiments.

The default experimental configuration uses one seed (42) and 50 communication rounds. Multiple seeds can be configured for more reliable comparisons.

## 9. Results and Outputs

The experiments generate performance metrics and visualizations, including:

- Global model accuracy and loss.
- Cumulative reward and episodic return.
- Success rate and convergence round.
- Client participation fairness.
- Training time and energy estimates.
- Communication cost estimates.
- RL decision and update time.
- Ablation study summaries.

## 10. Technologies Used

Python | PyTorch | Torchvision | NumPy | Pandas | Scikit-learn | Matplotlib | Seaborn | Reinforcement Learning | Federated Learning

## 11. Future Enhancements

- Evaluate the framework on additional datasets.
- Increase the number of communication rounds and experimental seeds.
- Evaluate additional adversarial attack scenarios.
- Validate simulated resource costs using real devices.
- Explore adaptive reward weighting.


