import xgboost as xgb
import lightgbm as lgb
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from data import preprocessing


#test dataset doesn't have labels, for checking model performance on test data
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_filename')
parser.add_argument('--model_filename')

args = parser.parse_args()


object = preprocessing.TrainDataset(dataset_name = args.dataset_filename)

X_val, y_val = object.preprocess(val = True)
model_lgb = lgb.Booster(model_file=r'../saved_models/lgb.txt')

pred_lgb = model_lgb.predict(X_val)
pred_lgb_labels = (pred_lgb > 0.4).astype(int)

print(pred_lgb_labels)

accuracy_lgb = accuracy_score(y_val, pred_lgb_labels)
f1_lgb = f1_score(y_val, pred_lgb_labels)
auc_lgb = roc_auc_score(y_val, pred_lgb_labels)
print('LightGBM')
print(f'accuracy:  {accuracy_lgb: .5f}')
print(f'f1 score:  {f1_lgb: .5f}')
print(f'auc score: {auc_lgb: .5f}')

model_xgb = xgb.Booster()
model_xgb.load_model('../saved_models/xgb.txt')

pred_xgb = model_xgb.predict(xgb.DMatrix(X_val))

pred_xgb_labels = (pred_xgb > 0.4).astype(int)


accuracy_xgb = accuracy_score(y_val, pred_xgb_labels)
f1_xgb = f1_score(y_val, pred_xgb_labels)
auc_xgb = roc_auc_score(y_val, pred_xgb_labels)
print('XGBoost')
print(f'accuracy:  {accuracy_xgb: .5f}')
print(f'f1 score:  {f1_xgb: .5f}')
print(f'auc score: {auc_xgb: .5f}')