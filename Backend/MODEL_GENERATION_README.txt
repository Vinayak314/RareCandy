HOW TO GENERATE AND USE BANK & CCP POLICY MODELS
=================================================

1. PREREQUISITES
----------------
Ensure you have the following dataset files in the `dataset/` folder:
- phase1_engineered_data.csv
- stocks_data_long.csv
- us_banks_interbank_matrix.csv

Ensure you have the required Python libraries installed:
- pandas
- numpy
- scikit-learn
- networkx

2. GENERATING THE MODELS
------------------------
Open a terminal in this directory (`RareCandy/ML`) and run:

    python final_model.py

This script will:
1. Initialize the CCP Game simulation.
2. Train the Bank and CCP agents for 1000 episodes (or as configured).
3. Save two files in the current directory:
   - `bank_policies.pkl`: A dictionary containing the trained MLPRegressor model for each bank.
   - `ccp_policy.pkl`: The trained MLPRegressor model for the Central Counterparty (CCP).

3. MOVING TO BACKEND
--------------------
Once the script finishes and you see "✅ Models saved..." in the output:

1. Create a `backend` folder if it doesn't exist (e.g., `RareCandy/backend/models`).
2. Move `bank_policies.pkl` and `ccp_policy.pkl` into this folder.

4. LOADING MODELS IN BACKEND
----------------------------
To use these models in your backend application:

```python
import pickle
import os

# Define path to your backend models folder
MODEL_DIR = "path/to/backend/models"

def load_models():
    # Load Bank Policies
    bank_model_path = os.path.join(MODEL_DIR, "bank_policies.pkl")
    with open(bank_model_path, "rb") as f:
        bank_policies = pickle.load(f)
        # bank_policies is a dict: { "BankName": <sklearn_model_object> }

    # Load CCP Policy
    ccp_model_path = os.path.join(MODEL_DIR, "ccp_policy.pkl")
    with open(ccp_model_path, "rb") as f:
        ccp_policy = pickle.load(f)
        # ccp_policy is a single sklearn model object

    return bank_policies, ccp_policy
```

5. TROUBLESHOOTING
------------------
- If `ModuleNotFoundError` occurs, ensure all dependencies are installed.
- If `FileNotFoundError` occurs during loading, check that the paths in `load_models` match where you moved the .pkl files.
