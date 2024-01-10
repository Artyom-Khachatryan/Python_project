import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class TrainDataset:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

    def read_dataset(self, dataset_name):
        data = pd.read_csv(dataset_name)
        return data
    
    def preprocess(self, val = False):
        data = self.read_dataset(self.dataset_name)
        
        X = data.drop(['min_ANNmuon', 'mass', 'production', 'signal', 'id', 'SPDhits'], axis=1)

        y = data['signal']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        scaler = MinMaxScaler()

        # Fit the scaler on the training data and transform it
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform the validation data using the same scaler
        X_val_scaled = scaler.transform(X_val)
        if val == False:
            return X_train_scaled, y_train
        elif val == True:
            return X_val_scaled, y_val        
      
class TestDataset:
    def __init__(self, dataset_name):

        self.dataset_name = dataset_name

    def read_dataset(self, dataset_name):
        data = pd.read_csv(dataset_name)
        return data
    
    def preprocess(self):
        data = self.read_dataset(self.dataset_name)
        
        X = data.drop(['id', 'SPDhits'], axis=1)

        scaler = MinMaxScaler()

        # Fit the scaler on the training data and transform it
        X_scaled = scaler.fit_transform(X)

        return X_scaled