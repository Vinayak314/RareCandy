# Neo-FinNet: Technical Task List
**Machine Learning for Systemic Risk & CCP Optimization in Financial Networks**

---

## Phase 1: Core Environment (The Sandbox)

### Module 1: Synthetic Network Generator
- [x] **1.1** Set up project structure and dependencies
  - [x] Initialize Python project with virtual environment
  - [x] Install core dependencies: `NetworkX`, `NumPy`, `Pandas`, `Matplotlib`
  - [x] Create project directory structure (`/src`, `/data`, `/models`, `/tests`, `/notebooks`)
  - [x] Set up version control (Git) with `.gitignore`

- [x] **1.2** Implement Bank/CCP Node class
  - [x] Define `Bank` class with attributes:
    - `bank_id`: Unique identifier
    - `equity`: Current equity capital
    - `assets`: Total assets
    - `liabilities`: Total liabilities
    - `liquidity`: Available liquid assets
    - `risk_weight`: Systemic importance score
  - [x] Implement methods: `update_balance_sheet()`, `calculate_leverage()`, `is_solvent()`
  - [x] Define `CCP` (Central Counterparty) class inheriting from `Bank` with additional attributes:
    - `margin_requirements`: Dictionary of margin rules
    - `default_fund`: Mutualized loss pool
    - `netting_positions`: Net positions per member
  - [x] Write unit tests for Bank and CCP classes

- [x] **1.3** Build Scale-Free Network Generator
  - [x] Implement Barabási-Albert preferential attachment algorithm
  - [x] Create `NetworkGenerator` class with methods:
    - `generate_scale_free(n_nodes, m_edges)`: Generate scale-free topology
    - `assign_node_attributes()`: Assign realistic financial attributes to nodes
    - `validate_network_properties()`: Check degree distribution follows power law
  - [x] Add configuration parameters (network density, clustering coefficient)
  - [x] Implement visualization function for network topology
  - [x] Write tests to verify scale-free properties (degree distribution, hub identification)

- [x] **1.4** Implement Interbank Loan Edge System
  - [x] Define `InterbankLoan` class with attributes:
    - `lender_id`, `borrower_id`
    - `principal`: Loan amount
    - `interest_rate`: Rate charged
    - `maturity`: Time to maturity
    - `collateral`: Pledged collateral value
  - [x] Create edge weight assignment based on bank size correlation
  - [x] Implement loan portfolio aggregation per bank
  - [x] Add method to calculate exposure concentration (Herfindahl index)

### Module 2: The CCP "Super-Node"

- [x] **2.1** Implement Multilateral Netting Engine
  - [x] Create `NettingEngine` class with method `calculate_net_positions(bilateral_trades)`
  - [x] Implement netting algorithm:
    - Input: List of bilateral trades between banks
    - Output: Single net position per bank vs. CCP
  - [x] Add compression ratio metric (bilateral vs. netted exposure)
  - [x] Write tests with sample trade datasets

- [x] **2.2** Build Margin Calculation System
  - [x] Implement `MarginCalculator` class with methods:
    - `calculate_initial_margin(position, volatility)`: Using SPAN or VaR methodology
    - `calculate_variation_margin(position, price_change)`: Mark-to-market adjustments
    - `calculate_default_fund_contribution(bank)`: Based on systemic importance
  - [x] Add configurable margin models (fixed percentage, VaR-based, SIMM)
  - [x] Implement margin call and collateral management logic
  - [x] Create visualization for margin requirements vs. bank liquidity

- [x] **2.3** Integrate CCP into Network
  - [x] Modify network structure to include CCP as central node
  - [x] Implement trade clearing workflow:
    - Bilateral trade submission → Netting → Margin posting → Settlement
  - [x] Add CCP default waterfall logic (margin → default fund → loss mutualization)
  - [x] Create metrics: total margin held, default fund size, CCP leverage

---

## Phase 2: Simulation Engine (The Game)

### Module 3: Agent Logic (Game Theory)

- [ ] **3.1** Define Bank Utility Function
  - [ ] Implement `BankAgent` class with utility function:
    - `U = α·ROE - β·Default_Risk - γ·Margin_Cost`
    - Parameters: `α` (profit weight), `β` (risk aversion), `γ` (liquidity cost)
  - [ ] Add decision-making methods:
    - `decide_lending(counterparty)`: Whether to lend based on expected utility
    - `optimize_portfolio()`: Adjust loan portfolio to maximize utility
  - [ ] Implement heterogeneous agent types (risk-averse, aggressive, neutral)

- [ ] **3.2** Implement Incomplete Information Framework
  - [ ] Create `InformationSet` class for each bank:
    - Own balance sheet (full visibility)
    - Direct counterparties (partial visibility)
    - CCP positions (visible)
    - Network topology (hidden)
  - [ ] Add noise/uncertainty to counterparty creditworthiness estimates
  - [ ] Implement Bayesian belief updating based on observed defaults

- [ ] **3.3** Build Strategic Interaction Engine
  - [ ] Implement game-theoretic decision loop:
    - Each bank observes its information set
    - Banks simultaneously choose lending/borrowing actions
    - Market clears, prices adjust
  - [ ] Add Nash equilibrium solver for static games (optional)
  - [ ] Create metrics: network formation dynamics, lending concentration

### Module 4: Shock & Cascade Simulator

- [x] **4.1** Design Shock Injection System
  - [x] Create `ShockGenerator` class with shock types:
    - `idiosyncratic_shock(bank_id, magnitude)`: Single bank equity loss
    - `systematic_shock(sector, magnitude)`: Correlated shock to bank subset
    - `liquidity_shock(market_rate)`: Sudden funding cost increase
  - [x] Implement shock severity distributions (uniform, normal, fat-tailed)
  - [x] Add temporal shock patterns (single event, repeated shocks, stress scenarios)

- [x] **4.2** Implement Contagion Propagation Algorithm
  - [x] Create `ContagionSimulator` class with method `propagate_default(initial_defaulter)`:
    - **Step 1**: Initial bank defaults → mark liabilities as losses
    - **Step 2**: Counterparties update balance sheets → check solvency
    - **Step 3**: If counterparty defaults → repeat from Step 1
    - **Step 4**: Terminate when no new defaults occur
  - [x] Add CCP-mediated contagion logic:
    - CCP absorbs losses via margin/default fund
    - If CCP fails → loss mutualization to surviving members
  - [x] Implement fire-sale externalities (asset liquidation depresses prices)

- [x] **4.3** Build Cascade Metrics & Analytics
  - [x] Implement metrics:
    - `cascade_size`: Number of defaulted banks
    - `cascade_depth`: Number of contagion rounds
    - `systemic_loss`: Total equity destroyed
    - `network_fragmentation`: Disconnected components post-cascade
  - [x] Create comparison metrics: with-CCP vs. without-CCP scenarios
  - [x] Add visualization: animated cascade propagation on network graph

- [ ] **4.4** Run Sensitivity Analysis
  - [ ] Vary parameters: network density, margin levels, shock size
  - [ ] Generate heatmaps: cascade size vs. (margin level, network density)
  - [ ] Identify critical thresholds (phase transitions in systemic risk)

---

## Phase 3: Machine Learning Layer (The Brain)

### Module 5: Data Pipeline

- [ ] **5.1** Design Simulation Experiment Framework
  - [ ] Create `ExperimentRunner` class to automate batch simulations:
    - Define parameter grid (interest rates, network density, margin levels, shock sizes)
    - Run N simulations per parameter combination
    - Store results in structured format
  - [ ] Implement parallel execution (multiprocessing/Dask)
  - [ ] Add progress tracking and checkpointing

- [ ] **5.2** Generate Training Dataset
  - [ ] Run 5,000+ simulations with varied parameters
  - [ ] For each simulation, capture snapshots at multiple timesteps:
    - `t=0`: Initial network state
    - `t=shock`: Immediately after shock
    - `t=1,2,...,T`: During cascade propagation
  - [ ] Extract features per snapshot:
    - **Node features**: equity, leverage, liquidity, degree centrality, betweenness
    - **Edge features**: loan size, interest rate, maturity
    - **Graph features**: clustering coefficient, average path length, assortativity
    - **CCP features**: margin coverage ratio, default fund size
  - [ ] Label each snapshot: `y=1` if cascade occurs, `y=0` if stable

- [ ] **5.3** Build Data Preprocessing Pipeline
  - [ ] Implement feature normalization (StandardScaler, MinMaxScaler)
  - [ ] Handle class imbalance (SMOTE, class weights)
  - [ ] Create train/validation/test split (70/15/15)
  - [ ] Convert NetworkX graphs to PyTorch Geometric `Data` objects
  - [ ] Save processed datasets to disk (HDF5 or PyTorch format)

### Module 6: Model Training

- [ ] **6.1** Implement Graph Neural Network Architecture
  - [ ] Set up PyTorch Geometric environment
  - [ ] Build GNN model (`GCN`, `GraphSAGE`, or `GAT`):
    - Input: Node features + Adjacency matrix
    - Hidden layers: 2-3 graph convolution layers
    - Output: Node embeddings
  - [ ] Add graph-level pooling (global mean/max pooling)
  - [ ] Implement final classifier: `MLP(graph_embedding) → P(crash)`

- [ ] **6.2** Train Binary Classifier (Crash Prediction)
  - [ ] Define loss function: Binary Cross-Entropy with class weights
  - [ ] Set up optimizer: Adam with learning rate scheduling
  - [ ] Implement training loop with validation monitoring
  - [ ] Add early stopping based on validation AUC-ROC
  - [ ] Track metrics: Accuracy, Precision, Recall, F1, AUC-ROC

- [ ] **6.3** Train Regression Model (Cascade Size Prediction)
  - [ ] Modify output layer to predict continuous `cascade_size`
  - [ ] Use MSE or Huber loss
  - [ ] Train and evaluate (R², MAE, RMSE)

- [ ] **6.4** Implement Explainability & Interpretability
  - [ ] Use GNNExplainer to identify critical nodes/edges
  - [ ] Visualize attention weights (if using GAT)
  - [ ] Perform ablation studies (remove features, measure performance drop)
  - [ ] Generate SHAP values for feature importance

- [ ] **6.5** Hyperparameter Tuning & Model Selection
  - [ ] Use Optuna or Ray Tune for hyperparameter search
  - [ ] Tune: learning rate, hidden dimensions, dropout, number of layers
  - [ ] Compare architectures: GCN vs. GraphSAGE vs. GAT
  - [ ] Select best model based on validation performance

---

## Phase 4: Business Impact & Visualization

### Module 7: The "Regulator Dashboard"

- [ ] **7.1** Set Up Visualization Framework
  - [ ] Choose framework: Streamlit (Python) or D3.js (JavaScript)
  - [ ] Create project structure for dashboard app
  - [ ] Set up backend API (Flask/FastAPI) to serve simulation results

- [ ] **7.2** Build Real-Time Network Visualization
  - [ ] Implement interactive network graph:
    - Nodes: Banks (size = equity, color = risk level)
    - Edges: Loans (thickness = exposure size)
    - CCP: Highlighted central node
  - [ ] Add dynamic updates during cascade simulation
  - [ ] Implement zoom, pan, node selection interactions

- [ ] **7.3** Create Risk Heatmap & Metrics Panel
  - [ ] Display key metrics:
    - System-wide leverage ratio
    - Network density & clustering
    - CCP margin coverage
    - Predicted crash probability (from ML model)
  - [ ] Add heatmap: systemic importance per bank (color-coded)
  - [ ] Implement alerts for high-risk configurations

- [ ] **7.4** Add Scenario Analysis Interface
  - [ ] Create controls to adjust parameters:
    - Shock size slider
    - Margin requirement slider
    - Network density selector
  - [ ] Run simulation on-demand with selected parameters
  - [ ] Display before/after comparison (with-CCP vs. without-CCP)

### Module 8: Optimization Report

- [ ] **8.1** Implement CCP Margin Optimization
  - [ ] Formulate optimization problem:
    - **Objective**: Minimize `P(systemic_crash)` (from ML model)
    - **Constraints**: Margin cost ≤ threshold, liquidity ≥ minimum
  - [ ] Use optimization library (SciPy, CVXPY, or genetic algorithms)
  - [ ] Run optimization across multiple network configurations

- [ ] **8.2** Generate Policy Recommendations
  - [ ] Create report template with sections:
    - **Current Risk Assessment**: Baseline crash probability
    - **Optimal Margin Levels**: Recommended margin rules
    - **Trade-offs**: Crash reduction vs. liquidity cost
    - **Sensitivity Analysis**: Robustness to parameter changes
  - [ ] Add visualizations: Pareto frontier (risk vs. cost)

- [ ] **8.3** Build Automated Report Generator
  - [ ] Implement PDF/HTML report generation (ReportLab, Jinja2)
  - [ ] Include:
    - Executive summary
    - Network topology diagrams
    - ML model predictions
    - Optimization results
    - Policy recommendations
  - [ ] Add export functionality (CSV, JSON for further analysis)

---

## Cross-Cutting Tasks

### Testing & Validation
- [ ] Write unit tests for all core modules (pytest)
- [ ] Implement integration tests for end-to-end simulation pipeline
- [ ] Validate against known financial network datasets (if available)
- [ ] Perform stress testing with extreme parameter values

### Documentation
- [ ] Write README with project overview and setup instructions
- [ ] Create API documentation (Sphinx or MkDocs)
- [ ] Document mathematical models and algorithms
- [ ] Add inline code comments and docstrings
- [ ] Create user guide for dashboard

### Performance Optimization
- [ ] Profile code to identify bottlenecks (cProfile)
- [ ] Optimize cascade algorithm (vectorization, Cython)
- [ ] Implement GPU acceleration for GNN training (CUDA)
- [ ] Add caching for repeated computations

### Deployment
- [ ] Containerize application (Docker)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Deploy dashboard to cloud (AWS/GCP/Azure)
- [ ] Implement logging and monitoring

---

## Suggested Implementation Order

**Week 1-2**: Phase 1 (Modules 1-2) - Build the financial network sandbox  
**Week 3-4**: Phase 2 (Modules 3-4) - Implement game theory and cascade simulation  
**Week 5-6**: Phase 3 (Module 5) - Generate training data  
**Week 7-8**: Phase 3 (Module 6) - Train ML models  
**Week 9-10**: Phase 4 (Modules 7-8) - Build dashboard and optimization  
**Week 11-12**: Testing, documentation, deployment

---

## Technology Stack Summary

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Graph Library | NetworkX, PyTorch Geometric |
| ML Framework | PyTorch |
| Optimization | SciPy, CVXPY |
| Visualization | Streamlit / D3.js |
| Data Storage | HDF5, PostgreSQL (optional) |
| Testing | pytest, unittest |
| Documentation | Sphinx, MkDocs |
| Deployment | Docker, FastAPI |

---

**Next Immediate Step**: Start with **Task 1.2** (Implement Bank/CCP Node class) to establish the core data structures.
