import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import xgboost as xgb

from evaluate import *



def train_model(data_path):
    df = pd.read_csv(data_path)

    # Encoding name features with target encoding
    nameEncoder = TargetEncoder(cols=['sire', 'dam', 'bmSire'], smoothing=10.0)

    y = df['rating']
    names = df['name']
    X = df.drop(columns=['rating', 'name'])
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42)

    X_train = nameEncoder.fit_transform(X_train, y_train)  
    X_test = nameEncoder.transform(X_test)                 


    # rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=13)
    xgbRegressor = xgb.XGBRegressor()
  
    # Fitting the model
    xgbRegressor.fit(X_train, y_train)


    # Evaluating the model with predctions on X_test
    y_pred = xgbRegressor.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    accuracy = xgbRegressor.score(X_test, y_test)
    
    print("\n==== Model Evaluation Metrics: ====")
    print(f"mse: {mse}")
    print(f"r2: {r2}")
    print(f"accuracy: {accuracy}")
    print("===================================\n")

    # display predictions
    display_predictions(xgbRegressor, X_test, y_test, names_test,  num_predictions=10)

    # display the decision tree
    # graph1 = xgb.to_graphviz(xgbRegressor, tree_idx=0)
    # graph1.render("xgb_tree 1")

    # graph2 = xgb.to_graphviz(xgbRegressor, tree_idx=50)
    # graph2.render("xgb_tree 50")

    
if __name__ == "__main__":
    data_path = "data/baseData.csv"
    train_model(data_path)