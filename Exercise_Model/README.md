Exercise Goal Prediction Model and FastAPI Deployment

Project Overview
This project is a comprehensive solution for predicting a person's fitness goal based on their personal and health-related attributes. It consists of two main parts: a machine learning pipeline that trains a deep learning model, and a FastAPI application that serves this model as a real-time prediction API.

The project addresses the challenge of making personalized fitness recommendations by leveraging a neural network trained on a dataset of individuals' health metrics.

Project Components
exercise_model.py: This script contains the entire machine learning pipeline. It handles data loading, preprocessing, model training, and saving all necessary components for deployment. This script should be run once to generate the model and preprocessor files.

exercise_model_deploye_fastapi.py: This script uses the FastAPI framework to create a web API. It loads the pre-trained model and preprocessing objects, and exposes a /predict endpoint that accepts user data, preprocesses it, and returns a fitness goal prediction.

Machine Learning Model
The core of this project is a deep learning model built with TensorFlow/Keras.

Data Preprocessing
The exercise_model.py script performs the following data preprocessing steps:

Outlier Removal: Outliers in numerical features (Age, Height, Weight, BMI) are removed using the Z-score method.

Categorical Encoding:

Label Encoding: Features like Sex, Hypertension, Diabetes, Fitness_Goal, and Fitness_Type are converted to numerical values.

One-Hot Encoding: The Level feature (e.g., 'Normal', 'Overweight') is transformed into a set of binary columns.

Feature Scaling: Numerical features are scaled using StandardScaler to ensure the model can learn efficiently.

Model Architecture
The model is a simple yet effective neural network:

An input layer to match the number of features after preprocessing.

Two Dense hidden layers with ReLU activation and l2 regularization to prevent overfitting.

A Dropout layer to further enhance regularization.

A Softmax output layer to provide a probability distribution over the possible fitness goals.

The model is trained using Stratified K-Fold cross-validation to ensure robust performance evaluation.

API Endpoints
The FastAPI application provides two endpoints:

1. GET /
Description: A root endpoint that returns a welcome message.

Response: A JSON object: {"message": "Welcome to Exercise Goal Prediction Model!"}

2. POST /predict
Description: This is the main endpoint for making predictions. It accepts a JSON payload with a user's health data and returns the predicted fitness goal.

Request Body:

{
  "Sex": "Male",
  "Age": 30,
  "Height": 175.0,
  "Weight": 70.0,
  "Hypertension": "No",
  "Diabetes": "No",
  "BMI": 22.86,
  "Level": "Normal",
  "Fitness_Type": "Cardio"
}

Response Body:

{
  "predicted_fitness_goal": "Weight Loss",
  "user_features": { ... }
}
