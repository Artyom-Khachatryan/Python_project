import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv('flavours-of-physics/training.csv')
data1 = pd.read_csv('flavours-of-physics/test.csv')
X = data.drop('signal', axis=1)
y = data['signal']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',  # Change to 'multiclass' for multiclass classification
    'metric': 'binary_logloss',  # Change to 'multi_logloss' for multiclass classification
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'is_unbalance': True,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

model = lgb.train(params, train_set = train_data)


y_pred = model.predict(X_val)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_pred_binary)
print(f"Accuracy: {accuracy:.4f}")