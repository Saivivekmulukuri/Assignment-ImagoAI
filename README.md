# DON Concentration Prediction Repository

This repository contains code and resources for predicting DON (vomitoxin) concentration in corn samples using hyperspectral imaging data. The project implements various machine learning approaches including Random Forest, XGBoost, Partial Least Squares (PLS) regression, and a PyTorch-based neural network. Additionally, a Streamlit app is provided for interactive predictions.

## Repository Structure

```markdown
.
├── data/
│   ├── TASK_ML_INTERN.csv        # Raw hyperspectral data file(s)
├── models/
│   ├── pls_model.pkl            # Saved PLS regression model (pickle file)
│   ├── nn_model.pt              # Saved PyTorch neural network model
│   ├── xgb_model.pkl            # Saved PyTorch XG Boost model
│   └── random_forest_model.pkl  # Saved Random Forest Regression model (pickle file)
├── notebooks/
│   └── Task.ipynb               # Jupyter Notebook for data exploration, modeling, and evaluation
├── streamlit_app/
│   ├── app.py                   # Streamlit app for interactive predictions from CSV uploads
│   └── selected_features.pkl    # Saved list of selected feature indices (pickle file)
├── requirements.txt             # Python package dependencies
└── README.md                    # Repository README file
```


## Installation and Running the Code

### 1. Install Dependencies

Make sure you have Python 3.7 or above installed. Then, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Alternatively, if you are using conda:

```bash
conda create -n don_prediction python=3.8
conda activate don_prediction
pip install -r requirements.txt
```

### 2. Running the Jupyter Notebook

The primary analysis and modeling code is located in the notebooks/Task.ipynb file. To launch the notebook:

```bash
jupyter notebook notebooks/Task_py.ipynb
```

This notebook includes all preprocessing, dimensionality reduction, model training (including Random Forest, XGBoost, PLS regression, and the neural network), and evaluation steps.

### 3. Running the Streamlit App
The Streamlit app is located in the ```streamlit_app``` directory and allows you to upload a CSV file and obtain DON concentration predictions. To run the app:

```bash
cd streamlit_app
streamlit run app.py
```
Only features with a correlation above a specified threshold (e.g., 0.15) are retained for modeling. The list of selected feature indices is stored in ```selected_features.pkl```.

### 4. Saved Models
The models/ directory contains the saved PLS model (pls_model.pkl), the neural network model (nn_model.pt), and the saved XGBoost model (xgb_model.pkl). These files can be loaded directly to perform predictions and evaluations.

### Note:
* Hyperparameter tuning is performed using GridSearchCV for Random Forest and XGBoost, while cross-validation is used to determine the optimal number of components in the PLS regression model.