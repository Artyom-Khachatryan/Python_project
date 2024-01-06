import pandas as pd
import lightgbm as lgb
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from data import preprocessing


parser = argparse.ArgumentParser()

parser.add_argument('filename')

args = parser.parse_args()

# Load the training data from CSV file
#this is valid link if you located in src folder
object = preprocessing.TrainDataset(args.filename)

X_train, y_train = object.preprocess()
X_val, y_val = object.preprocess(val = True)



train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Set hyperparameters for LightGBM
params = {
    'objective': 'binary',       # Binary classification task
    'metric': 'binary_logloss',  # Logarithmic loss (used for binary classification)
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'is_unbalance': True,         # Handle class imbalance
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,                # No output during training
}

# Train the LightGBM model
model = lgb.train(params, train_set=train_data, num_boost_round=300)

# Evaluate on the training set
y_pred = model.predict(X_train)
y_pred_binary = (y_pred > 0.4).astype(int)
accuracy = accuracy_score(y_train, y_pred_binary)
auc_score = roc_auc_score(y_train, y_pred_binary)
f1score = f1_score(y_train, y_pred_binary)

print(f"Train Accuracy: {accuracy:.4f}")

# Evaluate on the validation set
y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.4).astype(int)
accuracy = accuracy_score(y_val, y_pred_binary)
auc_score = roc_auc_score(y_val, y_pred_binary)
f1score = f1_score(y_val, y_pred_binary)

print(f"Val Accuracy: {accuracy:.4f}")

# Save the trained model
model.save_model(r'../saved_models/lgb.txt')

