import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import argparse
from data import preprocessing
#LightGBM

#test dataset doesn't have labels, for checking model performance on test data
parser = argparse.ArgumentParser()

parser.add_argument('filename')

args = parser.parse_args()


object = preprocessing.TestDataset(dataset_name = args.filename)

X = object.preprocess()

model = lgb.Booster(model_file=r'../saved_models/lgb.txt')

pred_lgb = model.predict(X)

pred_lgb_labels = (pred_lgb > 0.4).astype(int)

print(pred_lgb_labels)

#xgboost

model_xgb = xgb.Booster()
model_xgb.load_model('../saved_models/xgb.txt')

pred_xgb = model_xgb.predict(xgb.DMatrix(X))

pred_xgb_labels = (pred_xgb > 0.5).astype(int)

print(pred_xgb_labels)