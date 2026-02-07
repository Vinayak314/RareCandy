# Neo-FinNet

**Machine Learning for Systemic Risk & CCP Optimization in Financial Networks**

## Overview

Neo-FinNet is a graph-based simulation and machine learning system designed to model systemic risk in financial networks and optimize Central Counterparty (CCP) margin requirements. The project combines network science, game theory, and deep learning to predict and prevent financial contagion cascades.

## Features

- **Synthetic Financial Network Generation**: Create realistic scale-free banking networks
- **CCP Simulation**: Model multilateral netting and margin requirements
- **Contagion Cascades**: Simulate default propagation through interbank exposures
- **Graph Neural Networks**: Predict systemic crashes using GNN-based ML models
- **Regulatory Dashboard**: Visualize network risk and optimize CCP policies

## Project Structure

```
neo-finnet/
├── bank.py             # Bank and CCP classes
├── loan.py             # Interbank loans
├── network_generator.py # Scale-free network generation
├── test_bank.py        # Bank/CCP unit tests
├── test_loan.py        # Loan unit tests
├── test_network_generator.py # Network generator tests
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RareCandy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.core.bank import Bank, CCP
from src.network.generator import NetworkGenerator

# Create a financial network
generator = NetworkGenerator()
network = generator.generate_scale_free(n_nodes=50, m_edges=2)

# Add a CCP
ccp = CCP(ccp_id="CCP-1", equity=1000, margin_rate=0.05)

# Run simulation
# (Coming soon)
```

## Development Roadmap

- [x] Project setup and infrastructure
- [ ] Phase 1: Core Environment (Bank/CCP classes, network generation)
- [ ] Phase 2: Simulation Engine (Game theory, cascade simulation)
- [ ] Phase 3: ML Layer (GNN training, prediction models)
- [ ] Phase 4: Dashboard & Optimization

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License

## Contact

For questions or collaboration, please open an issue.
