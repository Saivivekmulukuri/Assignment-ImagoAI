# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Set page configuration and title
# st.set_page_config(page_title="DON Concentration Predictor", layout="wide")
# st.title("DON Concentration Predictor")
# st.write("""
# Upload a CSV file containing the sample features. The app will filter the features to those used 
# in training the model and output the predicted DON concentration for each sample.
# """)

# # Load the selected features list
# try:
#     with open("selected_features.pkl", "rb") as f:
#         selected_features = pickle.load(f)
#     st.success("Selected features loaded!")
# except Exception as e:
#     st.error(f"Error loading selected features: {e}")
#     selected_features = None

# # Load the pre-trained PLS regression model using pickle
# model_path = "../models/pls_model.pkl"
# try:
#     with open(model_path, "rb") as f:
#         pls_model = pickle.load(f)
#     st.success("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     pls_model = None

# # File uploader widget
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Read CSV file into a DataFrame
#     data = pd.read_csv(uploaded_file)
#     st.subheader("Uploaded Data")
#     st.dataframe(data.head())
#     data_feats = data.drop(columns=["hsi_id"])
#     # Check if the uploaded data has enough columns based on selected_features
#     selected_feat_nums = []
#     for feat in selected_features:
#         selected_feat_nums.append(int(feat))
#     if selected_features is None or data_feats.shape[1] < max(selected_feat_nums) + 1:
#         st.error("The uploaded file does not contain enough columns based on the selected features.")
#     else:
#         # Filter the DataFrame to keep only the columns used during training.
#         filtered_data = data_feats.iloc[:, selected_feat_nums]
#         st.subheader("Filtered Data (Selected Features)")
#         st.dataframe(filtered_data.head())

#         # If the model is loaded, perform predictions.
#         if pls_model is not None:
#             predictions = pls_model.predict(filtered_data)
#             # Convert predictions to a DataFrame for display.
#             pred_df = pd.DataFrame(predictions, columns=["Predicted DON Concentration"])
#             st.subheader("Predictions")
#             st.dataframe(pred_df)
#         else:
#             st.error("Model is not loaded; cannot perform predictions.")


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn

# -------------------------------
# Define the Neural Network Model
# -------------------------------
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize a fully-connected neural network for regression.
        
        Parameters:
        - input_dim (int): Number of input features.
        """
        super(RegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output layer for regression
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="DON Concentration Predictor", layout="wide")
st.title("DON Concentration Predictor")
st.write("""
Upload a CSV file containing the sample features. The app will filter the features to those used 
in training the neural network model and output the predicted DON concentration for each sample.
""")

# -------------------------------
# Load Selected Features List
# -------------------------------
try:
    with open("selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    st.success("Selected features loaded!")
except Exception as e:
    st.error(f"Error loading selected features: {e}")
    selected_features = None

# -------------------------------
# Load the Pre-trained Neural Network Model
# -------------------------------
# Define the path to your saved neural network model file.
model_path = "../models/nn_model.pt"
if selected_features is not None:
    # Assume that the number of selected features equals input dimension.
    input_dim = len(selected_features)
else:
    input_dim = 204  # Fallback value if selected_features is not loaded

# Instantiate the model and load the saved state dictionary.
try:
    model = RegressionNN(input_dim=input_dim)
    # Load the model state. Map to CPU to ensure compatibility.
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    st.success("Neural network model loaded successfully!")
except Exception as e:
    st.error(f"Error loading neural network model: {e}")
    model = None

# -------------------------------
# File Uploader Widget
# -------------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Remove non-feature columns (for example, assume "hsi_id" is not a feature)
    data_feats = data.drop(columns=["hsi_id"], errors='ignore')

    # Convert selected_features to integer indices if needed
    try:
        selected_feat_nums = [int(feat) for feat in selected_features]
    except Exception as e:
        st.error(f"Error processing selected features: {e}")
        selected_feat_nums = None

    if selected_feat_nums is None or data_feats.shape[1] < max(selected_feat_nums) + 1:
        st.error("The uploaded file does not contain enough columns based on the selected features.")
    else:
        # Filter the DataFrame to keep only the columns used during training.
        filtered_data = data_feats.iloc[:, selected_feat_nums]
        st.subheader("Filtered Data (Selected Features)")
        st.dataframe(filtered_data.head())

        # If the model is loaded, perform predictions.
        if model is not None:
            # Convert filtered data to a torch tensor of type float32.
            input_tensor = torch.tensor(filtered_data.values, dtype=torch.float32)
            with torch.no_grad():
                predictions = model(input_tensor).cpu().numpy()
            
            # Convert predictions to a DataFrame for display.
            pred_df = pd.DataFrame(predictions, columns=["Predicted DON Concentration"])
            st.subheader("Predictions")
            st.dataframe(pred_df)
        else:
            st.error("Model is not loaded; cannot perform predictions.")
