import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load Data
file_path = 'final_data.csv'
data = pd.read_csv(file_path)

# Encode Categorical Variables
categorical_cols = ['State', 'Drivetrain', 'Fuel type', 'Transmission', 'Engine']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Normalize Numeric Features
scaler = StandardScaler()
data[['Mileage', 'Year', 'Price(USD)']] = scaler.fit_transform(data[['Mileage', 'Year', 'Price(USD)']])

# Save Label Encoders and Scaler
if not os.path.exists('models'):
    os.makedirs('models')
for col, le in label_encoders.items():
    joblib.dump(le, f'models/label_encoder_{col}.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Feature Selection for Model Training
features = ['State', 'Drivetrain', 'Fuel type', 'Transmission', 'Engine', 'Mileage', 'Year', 'Price(USD)']
X = data[features]

# Apply Weights to Features
weights = {
    'Fuel type': 0.13,
    'Engine': 0.13,
    'Transmission': 0.13,
    'Drivetrain': 0.13,
    'Mileage': 0.065,
    'Year': 0.065,
    'Price(USD)': 0.2,
    'State': 0.15
}

# Create a weighted version of the dataset
weighted_X = X.copy()
for feature, weight in weights.items():
    weighted_X[feature] = weighted_X[feature] * weight

# Train Nearest Neighbors Model
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(weighted_X)
joblib.dump(model, 'models/nearest_neighbors_model.pkl')

# Streamlit User Inputs for Car Details
st.title('Car Recommendation System')
st.sidebar.header('Car Details')

def get_user_input():
    user_data = {}
    for col in ['State', 'Drivetrain', 'Fuel type', 'Transmission', 'Engine']:
        unique_values = list(label_encoders[col].classes_)
        user_value = st.sidebar.selectbox(f"Select {col}", options=unique_values, index=0)
        user_data[col] = label_encoders[col].transform([user_value])[0]

    user_data['Mileage'] = st.sidebar.number_input("Enter Mileage", min_value=0, value=10000, step=1000)
    user_data['Year'] = st.sidebar.number_input("Enter Year", min_value=1920, max_value=2025, value=2020, step=1)
    user_data['Price(USD)'] = st.sidebar.number_input("Enter Price (in USD)", min_value=0, value=20000, step=1000)

    return pd.DataFrame([user_data], columns=user_data.keys())

# Get User Input
df_user_input = get_user_input()

# Ensure all necessary features are provided
missing_cols = set(features) - set(df_user_input.columns)
for col in missing_cols:
    df_user_input[col] = 0  # Add missing columns with placeholder values

df_user_input = df_user_input[features]

# Normalize User Input if numeric columns are present
numeric_cols = ['Mileage', 'Year', 'Price(USD)']
df_user_input[numeric_cols] = scaler.transform(df_user_input[numeric_cols])

# Apply weights to user input
df_user_input_weighted = df_user_input.copy()
for feature, weight in weights.items():
    df_user_input_weighted[feature] = df_user_input_weighted[feature] * weight

# Load Nearest Neighbors Model
model = joblib.load('models/nearest_neighbors_model.pkl')

# Find Similar Cars
distances, indices = model.kneighbors(df_user_input_weighted)
recommended_cars = data.iloc[indices[0]]
recommended_cars['Similarity'] = 1 - distances[0]  # Similarity score (1 - distance)
recommended_cars = recommended_cars.sort_values(by='Similarity', ascending=False)

# Check if there are any recommended cars
if recommended_cars.empty:
    st.write('No recommendations found within the specified threshold.')
else:
    # Inverse Transform Categorical Columns
    for col in categorical_cols:
        recommended_cars[col] = label_encoders[col].inverse_transform(recommended_cars[col])

    # Inverse Transform Numeric Columns
    recommended_cars[['Mileage', 'Year', 'Price(USD)']] = scaler.inverse_transform(recommended_cars[['Mileage', 'Year', 'Price(USD)']])

    # Format Year column to display as an integer
    recommended_cars['Year'] = recommended_cars['Year'].astype(int)

    # Display Recommendations
    st.write('Top 5 Recommended Cars:')
    st.write(recommended_cars[['Make', 'Model', 'Year', 'Mileage', 'Price(USD)', 'Engine']].reset_index(drop=True))

    # Button to expand all details about recommended cars
    if st.button('Show Full Details'):
        st.write(recommended_cars.reset_index(drop=True))
