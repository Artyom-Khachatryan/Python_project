import pandas as pd
import lightgbm as lgb
test_data = pd.read_csv('dataset/flavours-of-physics/test.csv')

#test dataset doesn't have labels, for checking model performance on test data

model = lgb.Booster(model_file='saved_models/lgb.txt')
test_data = test_data.drop(['id', 'SPDhits'], axis=1)
pred = model.predict(test_data)
pred_labels = (pred > 0.4).astype(int)
print(pred_labels)
