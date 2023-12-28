import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score

training_data = pd.read_csv('dataset/flavours-of-physics/training.csv')

check_corellation = pd.read_csv('dataset/flavours-of-physics/check_correlation.csv')


X = training_data.drop(['min_ANNmuon', 'mass', 'production', 'signal', 'id', 'SPDhits'], axis=1)

y = training_data['signal']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)

params = {"objective": "binary:logistic",
          "eta": 0.4,
          "max_depth": 5,
          "min_child_weight": 3,
          "subsample": 0.5,
          "colsample_bytree": 0.7,
          "seed": 2}
n_trees = 100
model = xgb.train(params, xgb.DMatrix(X_train, y_train), n_trees)


y_pred = model.predict(xgb.DMatrix(X_train))
y_pred_binary = (y_pred > 0.4).astype(int)
accuracy = accuracy_score(y_train, y_pred_binary)
auc_score = roc_auc_score(y_train, y_pred_binary)
f1score = f1_score(y_train, y_pred_binary)

print('train')
print(f"Accuracy: {accuracy:.4f}")
print(f"Auc score: {auc_score:.4f}")
print(f"f1 score: {f1score:.4f}")
print(12*'*')

print('validation')
y_pred = model.predict(xgb.DMatrix(X_val))
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_pred_binary)
auc_score = roc_auc_score(y_val, y_pred_binary)
f1score = f1_score(y_val, y_pred_binary)

print(f"Accuracy: {accuracy:.4f}")
print(f"Auc score: {auc_score:.4f}")
print(f"f1 score: {f1score:.4f}")

#save model
model.save_model('saved_models/xgb.txt')

