import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error, r2_score



def train_model(data_path):
    df = pd.read_csv(data_path)

    # Encoding name features with target encoding
    nameEncoder = TargetEncoder(cols=['sire', 'dam', 'bmSire'], smoothing=10.0)

    y = df['rating']
    X = df.drop(columns=['rating', 'name'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = nameEncoder.fit_transform(X_train, y_train)  # fit on train
    X_test = nameEncoder.transform(X_test)                 # transform test (no fit!)


    rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)
    rf.fit(X_train, y_train)


    # Evaluating the model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"mse: {mse}")
    print(f"r2: {r2}")

if __name__ == "__main__":
    data_path = "data/baseData.csv"
    train_model(data_path)