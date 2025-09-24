import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import openpyxl
import pickle

# Load dataset
ex = pd.read_excel("gym recommendation.xlsx")
print(ex.head())
print(ex.dtypes)
print(ex.nunique())
print(ex.shape)
print(ex.isnull().sum())
print(ex["Fitness Type"].unique())

# Drop unnecessary columns
drop_cols = ["ID", "Exercises", "Equipment", "Diet", "Recommendation"]
ex.drop(columns= drop_cols, inplace=True)

# Remove outliers from numeric features using z-score method
outlaiers_num_cols = ["Age", "Height", "Weight", "BMI"]
z_score = np.abs(stats.zscore(ex[outlaiers_num_cols]))
exc = ex[(z_score < 3).all(axis= 1)]

print(f"Orginal_Data Shape = {ex.shape}\nData shape without outlaiers = {exc.shape}")

# Replace spaces in column names with underscores for easier handling
exc.columns = exc.columns.str.replace(" ", "_")

# Display unique categories in categorical columns
cat_cols = exc.select_dtypes(include= "object")
for col in cat_cols:
    print(f"{col} include {exc[col].nunique()}")

# Label encode categorical columns
label_cat_cols = ["Sex", "Hypertension", "Diabetes", "Fitness_Goal", "Fitness_Type"]
encoders = {}
for col in label_cat_cols:
    label = LabelEncoder()
    exc[col] = label.fit_transform(exc[col])
    encoders[col] = label   # Save each encoder for deployment

# One-hot encode "Level" column
ohe_cat_cols = ["Level"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_features = ohe.fit_transform(exc[ohe_cat_cols])
ohe_features_name = ohe.get_feature_names_out(ohe_cat_cols)
df_ohe_features = pd.DataFrame(ohe_features, columns= ohe_features_name, index= exc.index)

# Concatenate one-hot features with dataset
excr = pd.concat([exc.drop(columns= ohe_cat_cols), df_ohe_features], axis= 1)
print(excr.dtypes)

# Separate features and target variable
X = excr.drop(columns= "Fitness_Goal")
y = excr["Fitness_Goal"]

# Stratified K-Fold cross-validation setup
Skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
cv_train_acc = []
cv_test_acc = []

for train_index, test_index in Skfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale numeric features (fit on train, transform on test)
    train_num_columns = ["Age", "Height", "Weight", "BMI"]
    scaling = StandardScaler()
    X_train.loc[:, train_num_columns] = scaling.fit_transform(X_train[train_num_columns])
    X_test.loc[:, train_num_columns] = scaling.transform(X_test[train_num_columns])

    # Build deep learning model
    input_shape = X_train.shape[1]
    model = Sequential([
        Dense(64, activation= "relu", input_shape= (input_shape, ), kernel_regularizer= l2(0.001)),
        Dropout(0.5),
        Dense(32, activation= "relu", kernel_regularizer= l2(0.001)),
        Dense(y_train.nunique(), activation= "softmax")  # Output layer for classification
    ])

    # Compile model
    model.compile(optimizer= Adam(learning_rate= 0.0001), 
                  loss= "sparse_categorical_crossentropy", 
                  metrics= ["accuracy"])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor= "val_accuracy", patience= 10, restore_best_weights= True)

    # Train model
    model.fit(X_train, y_train, epochs= 1, batch_size= 32, validation_split= 0.2, callbacks= [early_stopping])

    # Predictions
    y_train_pred = np.argmax(model.predict(X_train), axis=1)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    cv_train_acc.append(train_acc)
    cv_test_acc.append(test_acc)

    fold += 1

# Display results
print(f"Train Accuracy = {cv_train_acc}")
print(f"Test Accuracy = {cv_test_acc}")
print(f"Train Average Accuracy = {np.mean(cv_train_acc)}")
print(f"Test Average Accuracy = {np.mean(cv_test_acc)}")

# Save model and preprocessing objects for deployment
model_filename = "exercise_model.keras"
label_filename = "label.pkl"
ohe_filename = "ohe.pkl"
scale_filename = "scale.pkl"

model.save(model_filename) 

with open(label_filename, "wb") as file:
    pickle.dump(encoders, file) 
with open(ohe_filename, "wb") as file:
    pickle.dump(ohe, file)
with open(scale_filename, "wb") as file:
    pickle.dump(scaling, file)     

print("All files saved successfully !")
