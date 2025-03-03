!pip install mlflow flask --no-cache-dir

############################################################################
# 0) INSTALLS (IF NEEDED)
############################################################################
# %pip install xgboost --quiet  # Uncomment if xgboost not installed

import os
import numpy as np
import pandas as pd
import cv2
import xgboost as xgb
import tensorflow as tf
import pickle
import joblib
import json
from datetime import datetime
import mlflow
import mlflow.xgboost
import mlflow.keras
import mlflow.sklearn
from flask import Flask, request, jsonify

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


############################################################################
# 1.1) MLFLOW TRACKING FUNCTIONS
############################################################################
def setup_mlflow():
    """Set up MLflow tracking"""
    # Set tracking URI to local directory in Kaggle
    mlflow.set_tracking_uri("file:///kaggle/working/mlruns")
    # Create experiment
    experiment_name = "vehicle_insurance_model"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_models_to_mlflow(clf_condition, reg_amount, extractor_model, metrics):
    """Log models and metrics to MLflow"""
    with mlflow.start_run(run_name="vehicle_insurance_model_run"):
        # Log parameters
        mlflow.log_param("clf_estimators", clf_condition.get_params()['n_estimators'])
        mlflow.log_param("clf_learning_rate", clf_condition.get_params()['learning_rate'])
        mlflow.log_param("clf_max_depth", clf_condition.get_params()['max_depth'])
        mlflow.log_param("reg_estimators", reg_amount.get_params()['n_estimators'])
        mlflow.log_param("reg_learning_rate", reg_amount.get_params()['learning_rate'])
        mlflow.log_param("reg_max_depth", reg_amount.get_params()['max_depth'])
        
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log models
        mlflow.xgboost.log_model(clf_condition, "condition_classifier")
        mlflow.xgboost.log_model(reg_amount, "amount_regressor")
        mlflow.keras.log_model(extractor_model, "feature_extractor")
        
        # Log preprocessing objects
        mlflow.sklearn.log_model(label_encoder_insurance, "label_encoder")
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
    return run_id


############################################################################
# 1) GLOBAL PREPROCESSING OBJECTS
############################################################################
label_encoder_insurance = LabelEncoder()
scaler = MinMaxScaler()

############################################################################
# 2) HELPER FUNCTIONS
############################################################################
def remove_outliers_iqr(df, columns, factor=1.0):
    """
    IQR outlier removal in specified columns.
    factor=1.0 => stricter than 1.5
    """
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df.reset_index(drop=True)

def oversample_condition_zero(df):
    """
    Duplicate Condition=0 rows to approach Condition=1 count.
    Encourages model to learn 0-labeled samples more.
    """
    if 'Condition' not in df.columns:
        return df
    df_0 = df[df['Condition'] == 0]
    df_1 = df[df['Condition'] == 1]
    if len(df_0) == 0 or len(df_1) == 0:
        return df
    count_0, count_1 = len(df_0), len(df_1)
    if count_0 < count_1:
        factor = count_1 // count_0
        remainder = count_1 % count_0
        df_0_oversample = pd.concat([df_0]*factor, ignore_index=True)
        df_0_oversample = pd.concat([df_0_oversample, df_0.sample(n=remainder, replace=True)], ignore_index=True)
        df = pd.concat([df_0_oversample, df_1], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def fill_and_scale_tabular(df, is_train=True):
    """
    1) Extract y_cond, y_amt from df
    2) Drop them so they're NOT used in X
    3) Fill missing numeric
    4) Encode 'insurance'
    5) Scale numeric columns except Amount
    Returns: X_tab, y_cond, y_amt
    """
    # Condition & Amount => targets
    y_cond = df['Condition'].values if 'Condition' in df.columns else None
    y_amt  = df['Amount'].values    if 'Amount' in df.columns else None

    # Drop them from df so they do NOT appear in numeric features
    df.drop(["Condition", "Amount"], axis=1, inplace=True, errors="ignore")

    # If 'insurance' missing
    if 'insurance' not in df.columns:
        df['insurance'] = 'unknown'
    df['insurance'] = df['insurance'].fillna('unknown').astype(str)

    # Encode 'insurance'
    if is_train:
        label_encoder_insurance.fit(df['insurance'])
    else:
        df['insurance'] = df['insurance'].apply(
            lambda x: x if x in label_encoder_insurance.classes_ else 'unknown'
        )
        if 'unknown' not in label_encoder_insurance.classes_:
            label_encoder_insurance.classes_ = np.append(label_encoder_insurance.classes_, 'unknown')
    df['insurance'] = label_encoder_insurance.transform(df['insurance'])

    # numeric_cols
    numeric_cols = ['insurance','Cost_of_vehicle','Min_coverage','Max_coverage']

    # Fill missing
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median() if not df[col].dropna().empty else 0
        df[col] = df[col].fillna(median_val)

    # Expiry_date -> ordinal
    if 'Expiry_date' in df.columns:
        df['Expiry_date'] = pd.to_datetime(df['Expiry_date'], errors='coerce')
        df['expiry_date_ordinal'] = df['Expiry_date'].apply(lambda d: d.toordinal() if not pd.isnull(d) else 0)
    else:
        df['expiry_date_ordinal'] = 0

    numeric_cols += ['expiry_date_ordinal']

    # Scale
    if is_train:
        scaler.fit(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    X_tab = df[numeric_cols].values
    return X_tab, y_cond, y_amt

def build_feature_extractor():
    """
    Build a pretrained EfficientNetB0 for feature extraction (frozen).
    Returns model that outputs a 1280-dim feature vector for each image.
    """
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=x)
    # Freeze all
    for layer in model.layers:
        layer.trainable = False
    return model

def load_images_and_extract_features(df, folder, extractor_model, log_transform=False, is_train=True):
    """
    1) fill_and_scale_tabular => X_tab, y_cond, y_amt
    2) Log transform y_amt if needed
    3) For each image => pass to extractor => feats(1280)
    4) Concatenate feats + X_tab => X_all
    """
    X_tab, y_cond, y_amt = fill_and_scale_tabular(df, is_train=is_train)

    # Log transform if requested
    if log_transform and y_amt is not None:
        y_amt = np.clip(y_amt, 0, None)
        y_amt = np.log1p(y_amt)

    features_list = []
    for img_file in df['Image_path']:
        path = os.path.join(folder, img_file)
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)
        img = cv2.resize(img, (224,224))
        # BGR->RGB
        img = img[:, :, ::-1]
        # [0,1]
        img = img.astype('float32') / 255.0
        # Preprocess for EfficientNet
        img = preprocess_input(img * 255.0)
        img_batch = np.expand_dims(img, axis=0)  # (1,H,W,C)
        feats = extractor_model.predict(img_batch, verbose=0)
        features_list.append(feats[0])  # shape (1280,)

    X_img_features = np.array(features_list, dtype=np.float32)
    X_all = np.hstack([X_img_features, X_tab])
    return X_all, y_cond, y_amt

# New serialization functions
def serialize_models(clf_condition, reg_amount, extractor_model, output_dir="serialized_models"):
    """
    Serialize all models and preprocessing objects to disk.
    Returns a dictionary with paths to all serialized files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}
    
    # 1. Save condition classifier
    condition_path = os.path.join(output_dir, f"clf_condition_{timestamp}.json")
    clf_condition.save_model(condition_path)
    paths["condition_model"] = condition_path
    
    # 2. Save amount regressor
    amount_path = os.path.join(output_dir, f"reg_amount_{timestamp}.json")
    reg_amount.save_model(amount_path)
    paths["amount_model"] = amount_path
    
    # 3. Save feature extractor (TensorFlow model)
    # Add .keras extension to the file path
    extractor_path = os.path.join(output_dir, f"extractor_{timestamp}.keras")
    extractor_model.save(extractor_path)
    paths["extractor_model"] = extractor_path
    
    # 4. Save label encoder
    encoder_path = os.path.join(output_dir, f"label_encoder_{timestamp}.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder_insurance, f)
    paths["label_encoder"] = encoder_path
    
    # 5. Save scaler
    scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    paths["scaler"] = scaler_path
    
    # 6. Save metadata (info about the models)
    metadata = {
        "timestamp": timestamp,
        "files": paths,
        "features": {
            "tabular_features": ['insurance', 'Cost_of_vehicle', 'Min_coverage', 'Max_coverage', 'expiry_date_ordinal'],
            "image_feature_size": 1280
        },
        "preprocessing": {
            "image_size": [224, 224],
            "log_transform_amount": True
        }
    }
    
    metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    paths["metadata"] = metadata_path
    
    return paths
    
def load_serialized_models(model_dir, timestamp=None):
    """
    Load serialized models from disk.
    If timestamp is None, loads the latest models.
    Returns dict with loaded models and preprocessing objects.
    """
    # Find latest timestamp if not provided
    if timestamp is None:
        metadata_files = [f for f in os.listdir(model_dir) if f.startswith("metadata_")]
        if not metadata_files:
            raise FileNotFoundError("No metadata files found in the directory.")
        
        # Extract the full timestamp (fix: don't split on '.' yet)
        timestamps = []
        for f in metadata_files:
            # Extract everything between "metadata_" and ".json"
            ts = f.replace("metadata_", "").replace(".json", "")
            timestamps.append(ts)
        
        timestamp = max(timestamps)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load models
    models = {}
    
    # 1. Load condition classifier
    clf_path = metadata["files"]["condition_model"]
    models["clf_condition"] = xgb.XGBClassifier()
    models["clf_condition"].load_model(clf_path)
    
    # 2. Load amount regressor
    reg_path = metadata["files"]["amount_model"]
    models["reg_amount"] = xgb.XGBRegressor()
    models["reg_amount"].load_model(reg_path)
    
    # 3. Load feature extractor
    extractor_path = metadata["files"]["extractor_model"]
    models["extractor_model"] = tf.keras.models.load_model(extractor_path)
    
    # 4. Load label encoder
    encoder_path = metadata["files"]["label_encoder"]
    with open(encoder_path, 'rb') as f:
        models["label_encoder"] = pickle.load(f)
    
    # 5. Load scaler
    scaler_path = metadata["files"]["scaler"]
    with open(scaler_path, 'rb') as f:
        models["scaler"] = pickle.load(f)
    
    models["metadata"] = metadata
    return models

def make_prediction_from_serialized(models, image_path, tabular_data):
    """
    Make predictions using loaded serialized models.
    
    Args:
        models: Dictionary of loaded models from load_serialized_models()
        image_path: Path to the vehicle image
        tabular_data: DataFrame with single row of tabular data
    
    Returns:
        Tuple of (condition_prediction, amount_prediction)
    """
    # Get models
    extractor_model = models["extractor_model"]
    clf_condition = models["clf_condition"]
    reg_amount = models["reg_amount"]
    
    # Set global preprocessing objects
    global label_encoder_insurance, scaler
    label_encoder_insurance = models["label_encoder"]
    scaler = models["scaler"]
    
    # Make prediction
    # Create a DataFrame with the correct structure
    df = tabular_data.copy()
    df["Image_path"] = os.path.basename(image_path)
    
    # Create a temporary directory for the image
    temp_folder = os.path.dirname(image_path)
    
    # Extract features
    X_all, _, _ = load_images_and_extract_features(
        df=df,
        folder=temp_folder,
        extractor_model=extractor_model,
        log_transform=False,
        is_train=False
    )
    
    # Predict condition
    cond_prob = clf_condition.predict_proba(X_all)[:,1][0]
    condition_pred = 1 if cond_prob > 0.4 else 0
    
    # Predict amount
    amount_log = reg_amount.predict(X_all)[0]
    amount_pred = np.expm1(amount_log)
    
    return condition_pred, amount_pred, cond_prob

############################################################################
# 3) MAIN CODE
############################################################################
# File paths
train_csv = "/kaggle/input/dataset1/Fast_Furious_Insured/train.csv"
test_csv  = "/kaggle/input/dataset1/Fast_Furious_Insured/test.csv"
train_imgs_folder = "/kaggle/input/dataset1/Fast_Furious_Insured/trainImages"
test_imgs_folder  = "/kaggle/input/dataset1/Fast_Furious_Insured/testImages"

# 3.1 Read train
train_df = pd.read_csv(train_csv)

# 3.2 Oversample Condition=0
train_df = oversample_condition_zero(train_df)

# 3.3 Remove outliers
cols_outliers = ['Cost_of_vehicle','Min_coverage','Max_coverage','Amount']
train_df = remove_outliers_iqr(train_df, cols_outliers, factor=1.0)

print("After oversampling + outlier removal =>", train_df.shape)

# 3.4 Build feature extractor
extractor_model = build_feature_extractor()

# 3.5 Build arrays for train (log transform Amount)
X_all, y_cond, y_amt_log = load_images_and_extract_features(
    df=train_df,
    folder=train_imgs_folder,
    extractor_model=extractor_model,
    log_transform=True,
    is_train=True
)

# 3.6 Train/Val Split
from sklearn.model_selection import train_test_split
X_train, X_val, cond_train, cond_val, amt_train_log, amt_val_log = train_test_split(
    X_all, y_cond, y_amt_log, test_size=0.2, random_state=42
)

print("Train shape =>", X_train.shape, cond_train.shape, amt_train_log.shape)
print("Val shape   =>", X_val.shape, cond_val.shape, amt_val_log.shape)

############################################################################
# 4) XGBOOST FOR CONDITION
############################################################################
ratio_0 = (cond_train == 0).sum()
ratio_1 = (cond_train == 1).sum()
if ratio_0 == 0 or ratio_1 == 0:
    scale_pos_weight = 1.0
else:
    scale_pos_weight = ratio_0 / ratio_1  # #neg/#pos

clf_condition = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist'  # or 'hist' if no GPU
)

clf_condition.fit(
    X_train, cond_train,
    eval_set=[(X_val, cond_val)],
    early_stopping_rounds=20,
    verbose=True
)

cond_val_pred = clf_condition.predict(X_val)
acc = accuracy_score(cond_val, cond_val_pred)
print("Validation Condition Accuracy:", acc)

############################################################################
# 5) XGBOOST FOR AMOUNT (REGRESSION)
############################################################################
reg_amount = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    tree_method='gpu_hist'  # or 'hist'
)

reg_amount.fit(
    X_train, amt_train_log,
    eval_set=[(X_val, amt_val_log)],
    early_stopping_rounds=20,
    verbose=True
)

amt_val_pred_log = reg_amount.predict(X_val)
amt_val_pred = np.expm1(amt_val_pred_log)  # invert log
amt_val_real = np.expm1(amt_val_log)

mse_ = mean_squared_error(amt_val_real, amt_val_pred)
rmse_ = np.sqrt(mse_)
mae_ = mean_absolute_error(amt_val_real, amt_val_pred)
print("\nValidation Amount MSE:", mse_)
print("Validation Amount RMSE:", rmse_)
print("Validation Amount MAE:", mae_)

############################################################################
# 6) INFERENCE ON TEST
############################################################################
test_df = pd.read_csv(test_csv)
X_test_all, _, _ = load_images_and_extract_features(
    df=test_df,
    folder=test_imgs_folder,
    extractor_model=extractor_model,
    log_transform=False,
    is_train=False
)

test_cond_raw = clf_condition.predict_proba(X_test_all)[:,1]
threshold_cond = 0.4
test_cond_pred = (test_cond_raw > threshold_cond).astype(int)

test_amt_log = reg_amount.predict(X_test_all)
test_amt_pred = np.expm1(test_amt_log)

out_df = pd.DataFrame({
    "Image_path": test_df["Image_path"],
    "Condition_prob_of_1": test_cond_raw,
    "Condition_pred": test_cond_pred,
    "Amount_pred": test_amt_pred
})

# Save final output in test_predictionsopt.csv
out_df.to_csv("test_predictionsopt196.csv", index=False)
print("\nTest predictions saved to 'test_predictionsopt196.csv'.")
print(out_df.head(10))



############################################################################
# 5.1) LOG MODELS TO MLFLOW
############################################################################
# Setup MLflow
experiment_id = setup_mlflow()

# Prepare metrics dictionary
metrics = {
    "condition_accuracy": accuracy_score(cond_val, cond_val_pred),
    "amount_mse": mean_squared_error(amt_val_real, amt_val_pred),
    "amount_rmse": np.sqrt(mean_squared_error(amt_val_real, amt_val_pred)),
    "amount_mae": mean_absolute_error(amt_val_real, amt_val_pred)
}

# Log models to MLflow
mlflow_run_id = log_models_to_mlflow(clf_condition, reg_amount, extractor_model, metrics)
print(f"Models logged to MLflow with run ID: {mlflow_run_id}")


############################################################################
# 7) MODEL SERIALIZATION
############################################################################
# Create serialized model directory
serialized_dir = "serialized_models"
if not os.path.exists(serialized_dir):
    os.makedirs(serialized_dir)

# Serialize models
print("\n\nSerializing models...")
serialized_paths = serialize_models(
    clf_condition=clf_condition,
    reg_amount=reg_amount,
    extractor_model=extractor_model,
    output_dir=serialized_dir
)

# Print serialization information
print("\nModels successfully serialized!")
print("Serialized files:")
for model_type, path in serialized_paths.items():
    print(f"- {model_type}: {path}")

# Test loading the models
print("\nTesting model loading...")
loaded_models = load_serialized_models(serialized_dir)
print("Models successfully loaded!")

# Demonstration of how to use the loaded models for a single prediction
print("\nDemonstration of prediction with loaded models:")
# Use the first test image as an example
sample_img_path = os.path.join(test_imgs_folder, test_df["Image_path"].iloc[0])
sample_data = test_df.iloc[[0]].copy()  # Get the first row as a DataFrame

# Make prediction with loaded models
try:
    condition, amount, prob = make_prediction_from_serialized(
        loaded_models, 
        sample_img_path, 
        sample_data
    )
    print(f"Sample prediction:")
    print(f"- Image: {test_df['Image_path'].iloc[0]}")
    print(f"- Condition: {condition} (probability: {prob:.4f})")
    print(f"- Amount: ${amount:.2f}")
    
    # Compare with original prediction
    print(f"\nOriginal prediction:")
    print(f"- Condition: {test_cond_pred[0]} (probability: {test_cond_raw[0]:.4f})")
    print(f"- Amount: ${test_amt_pred[0]:.2f}")
    
except Exception as e:
    print(f"Error in demonstration: {e}")

# Print list of all serialized files for reference
print("\nComplete list of serialized files:")
for root, dirs, files in os.walk(serialized_dir):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"- {file_path} ({file_size:.2f} KB)")

# Log serialized model paths to MLflow
with mlflow.start_run(run_id=mlflow_run_id):
    for model_type, path in serialized_paths.items():
        mlflow.log_artifact(path, "serialized_models")
    print(f"\nSerialized models logged to MLflow run {mlflow_run_id}")



############################################################################
# 8) FLASK API FOR PREDICTIONS
############################################################################
# Define API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract data
    image_path = data.get('image_path')
    tabular_data = pd.DataFrame([data.get('tabular_data')])
    
    # Load models (using your existing function)
    models = load_serialized_models("serialized_models")
    
    # Make prediction
    try:
        condition, amount, prob = make_prediction_from_serialized(
            models, 
            image_path, 
            tabular_data
        )
        
        return jsonify({
            "condition": int(condition),
            "condition_probability": float(prob),
            "amount": float(amount)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to save the Flask app to a file for deployment
def save_flask_app():
    with open('/kaggle/working/app.py', 'w') as f:
        f.write("""
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import xgboost as xgb
import pickle
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Copy these functions from your notebook
""" + inspect.getsource(remove_outliers_iqr) + "\n" +
        inspect.getsource(fill_and_scale_tabular) + "\n" +
        inspect.getsource(load_serialized_models) + "\n" +
        inspect.getsource(make_prediction_from_serialized) + """

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
        
    # Save uploaded image
    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)
    
    # Get tabular data
    tabular_data = json.loads(request.form.get('data', '{}'))
    tabular_df = pd.DataFrame([tabular_data])
    
    # Load models
    models = load_serialized_models("serialized_models")
    
    # Make prediction
    try:
        condition, amount, prob = make_prediction_from_serialized(
            models, 
            image_path, 
            tabular_df
        )
        
        return jsonify({
            "condition": int(condition),
            "condition_probability": float(prob),
            "amount": float(amount)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
""")
    print("Flask app saved to /kaggle/working/app.py")

# Create API file
import inspect
save_flask_app()

# Test API within notebook
def test_api():
    # Use the first test image as an example
    sample_img_path = os.path.join(test_imgs_folder, test_df["Image_path"].iloc[0])
    sample_data = test_df.iloc[0].to_dict()
    
    # Create test request
    test_request = {
        "image_path": sample_img_path,
        "tabular_data": {
            "insurance": sample_data["insurance"] if "insurance" in sample_data else "unknown",
            "Cost_of_vehicle": sample_data["Cost_of_vehicle"] if "Cost_of_vehicle" in sample_data else 0,
            "Min_coverage": sample_data["Min_coverage"] if "Min_coverage" in sample_data else 0,
            "Max_coverage": sample_data["Max_coverage"] if "Max_coverage" in sample_data else 0,
            "Expiry_date": sample_data["Expiry_date"] if "Expiry_date" in sample_data else None
        }
    }
    
    # Test the function directly (bypassing HTTP)
    models = load_serialized_models("serialized_models")
    condition, amount, prob = make_prediction_from_serialized(
        models, 
        test_request["image_path"], 
        pd.DataFrame([test_request["tabular_data"]])
    )
    
    print("API Test Result:")
    print(f"- Condition: {condition} (probability: {prob:.4f})")
    print(f"- Amount: ${amount:.2f}")
    
    return {
        "condition": int(condition),
        "condition_probability": float(prob),
        "amount": float(amount)
    }

# Run test
test_result = test_api()
print(f"Test API Response: {json.dumps(test_result, indent=2)}")



############################################################################
# 9) DEPLOYMENT INSTRUCTIONS
############################################################################
print("""
To deploy the API:
1. Download the serialized_models directory from this Kaggle notebook
2. Download the app.py file
3. Install requirements: pip install flask tensorflow xgboost scikit-learn opencv-python numpy pandas
4. Run the API: python app.py
5. Example API call:
import requests
import json
# API endpoint
url = 'http://localhost:8000/predict'
# Prepare data
files = {'image': open('C:\\Users\\pbtra\\Desktop\\sdzfd.jpg', 'rb')}
data = {
    'data': json.dumps({
        'insurance': 'unknown',
        'Cost_of_vehicle': 25000,
        'Min_coverage': 10000,
        'Max_coverage': 50000,
        'Expiry_date': '2025-12-31'
    })
}
# Send request
response = requests.post(url, files=files, data=data)
print(response.json())
""")
# Create a metadata file for deployment
deployment_metadata = {
    "mlflow_run_id": mlflow_run_id,
    "serialized_model_directory": "serialized_models",
    "api_file": "app.py",
    "required_packages": [
        "flask", 
        "tensorflow", 
        "xgboost", 
        "scikit-learn", 
        "opencv-python", 
        "numpy", 
        "pandas",
        "mlflow"
    ],
    "api_endpoint": "/predict",
    "input_format": {
        "image": "multipart file upload",
        "data": "JSON with insurance, Cost_of_vehicle, Min_coverage, Max_coverage, Expiry_date"
    },
    "output_format": {
        "condition": "int (0 or 1)",
        "condition_probability": "float between 0 and 1",
        "amount": "float"
    }
}
with open('/kaggle/working/deployment_metadata.json', 'w') as f:
    json.dump(deployment_metadata, f, indent=4)
print("Deployment metadata saved to /kaggle/working/deployment_metadata.json")
