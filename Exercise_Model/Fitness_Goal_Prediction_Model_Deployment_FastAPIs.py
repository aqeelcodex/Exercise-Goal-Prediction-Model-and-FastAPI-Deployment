from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize FastAPI app with metadata
app = FastAPI(
    title= "Deployed a Crop Yield Prediction model using FastAPI",
    description= "Deployed a Crop Yield Prediction model using FastAPI. The project integrates machine learning with a scalable API for real-time predictions, enabling farmers and researchers to estimate crop yield efficiently with accessible endpoints.",
    version= "1.0.0"
)

# Load trained model and preprocessing objects
try:
    model = load_model("exercise_model.keras")  # Load saved Keras model
    with open("label.pkl", "rb") as file:       # Load label encoders for categorical variables
        label_loaded = pickle.load(file)
    with open("ohe.pkl", "rb") as file:         # Load OneHotEncoder for categorical features
        ohe_loaded = pickle.load(file)
    with open("scale.pkl", "rb") as file:       # Load StandardScaler for numerical features
        scale_loaded = pickle.load(file)
except FileNotFoundError as e:
    print(f"Some file is not Found: {e}")

# Input schema for request body
class ExerciseInputs(BaseModel):
    """
    A class to represent the input features for the fitness goal prediction model.
    """
    Sex: str
    Age: int
    Height: float
    Weight: float
    Hypertension: str
    Diabetes: str
    BMI: float
    Level: str
    Fitness_Type: str

# Define response schema for consistent API output
class PredictionResponse(BaseModel):
    predicted_value: str
    crop_features: dict

# Feature groups
label_enc_cols = ["Sex", "Hypertension", "Diabetes", "Fitness_Type"]  # Categorical features (LabelEncoder)
ohe_cat_cols = ["Level"]                                             # Categorical features (OneHotEncoder)
num_cols = ["Age", "Height", "Weight", "BMI"]                        # Numerical features (Scaler)

# Function to preprocess input before prediction
def preprocess_input(data: dict):
    try:
        # Create dataframe from input dictionary
        df = pd.DataFrame([data])

        # Apply label encoding to categorical features
        for col in label_enc_cols:
            le = label_loaded[col]
            df[col] = le.transform(df[col])

        # Apply one-hot encoding to categorical features
        ohe_features = ohe_loaded.transform(df[ohe_cat_cols])
        ohe_features_name = ohe_loaded.get_feature_names_out(ohe_cat_cols)
        ohe_df = pd.DataFrame(ohe_features, columns=ohe_features_name, index=df.index)

        # Merge one-hot encoded features with other columns
        processed_df = pd.concat([df.drop(columns=ohe_cat_cols), ohe_df], axis=1)

        # Scale numerical features
        processed_df[num_cols] = scale_loaded.transform(processed_df[num_cols])

        # Convert final dataframe to numpy array for model input
        return processed_df.to_numpy(dtype=np.float32)

    except KeyError as e:
        # If input is missing any required key
        raise HTTPException(
            status_code=400,
            detail=f"Missing a required key in input data: {str(e)}"
        )
    except ValueError as e:
        # If encoding/scaling fails due to invalid values
        raise HTTPException(
            status_code=500,
            detail=f"Encoding error: {str(e)}"
        )

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Exercise Goal Prediction Model!"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def prediction_func(exercise: ExerciseInputs):
    try:
        # Preprocess input
        model_input = preprocess_input(exercise.dict())

        # Debugging logs (can be removed later)
        print("Model input type:", type(model_input))
        print("Model input dtype:", model_input.dtype)
        print("Model input shape:", model_input.shape)
        print("Model input values:", model_input)

        # Predict probabilities for each class
        prediction_probs = model.predict(model_input)

        # Get index of the highest probability class
        prediction_index = np.argmax(prediction_probs, axis= 1)[0]

        # Convert numeric prediction back to original class label
        final_prediction = label_loaded["Fitness_Goal"].inverse_transform([prediction_index])[0]

        # Return structured response
        return PredictionResponse(
            predicted_value= final_prediction,
            crop_features= exercise.dict()
        )
    except Exception as e:
       # Catch-all error handling
       raise HTTPException(status_code=500, detail= f"Something is going wrong. {str(e)}")
    
# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host= "127.0.0.1", port= 8000)
