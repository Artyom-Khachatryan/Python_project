import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# Load the training data from CSV file
training_data = pd.read_csv('dataset/flavours-of-physics/training.csv')

# Drop columns that are not used as features
X = training_data.drop(['min_ANNmuon', 'mass', 'production', 'signal', 'id', 'SPDhits'], axis=1)

# Target variable
y = training_data['signal']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Create LightGBM datasets for training and validation
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Set hyperparameters for LightGBM
params = {
    'objective': 'binary',       # Binary classification task
    'metric': 'binary_logloss',  # Logarithmic loss (used for binary classification)
    'boosting_type': 'gbdt',
    'num_leaves': 35,
    'is_unbalance': True,         # Handle class imbalance
    'learning_rate': 0.15,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,                # No output during training
}

# Train the LightGBM model
model = lgb.train(params, train_set=train_data, num_boost_round=200)

# Evaluate on the training set
y_pred = model.predict(X_train)
y_pred_binary = (y_pred > 0.4).astype(int)
accuracy = accuracy_score(y_train, y_pred_binary)
auc_score = roc_auc_score(y_train, y_pred_binary)
f1score = f1_score(y_train, y_pred_binary)

print('Training Metrics:')
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print(f"F1 Score: {f1score:.4f}")
print(12 * '*')

# Evaluate on the validation set
y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.4).astype(int)
accuracy = accuracy_score(y_val, y_pred_binary)
auc_score = roc_auc_score(y_val, y_pred_binary)
f1score = f1_score(y_val, y_pred_binary)

print('Validation Metrics:')
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print(f"F1 Score: {f1score:.4f}")

# Save the trained model
model.save_model('saved_models/lgb.txt')

