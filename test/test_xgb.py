import pandas as pd
import xgboost as xgb
test_data = pd.read_csv('dataset/flavours-of-physics/test.csv')
#test dataset doesn't have labels, for checking model performance on test data
model = xgb.Booster()
model.load_model('saved_models/xgb.txt')
test_data = test_data.drop(['id', 'SPDhits'], axis=1)
pred = model.predict(xgb.DMatrix(test_data))
pred_labels = (pred > 0.4).astype(int)
print(pred_labels)
