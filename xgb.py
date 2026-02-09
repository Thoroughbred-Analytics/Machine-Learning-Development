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

"""
Helper function to display the validation of the model during training.
"""
def graph_training(model):
    results = model.evals_result()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    # Logloss
    ax1.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax1.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax1.legend()
    ax1.set_ylabel('Log Loss')
    ax1.set_xlabel('Boosting Round')
    ax1.set_title('XGBoost Log Loss')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def train_model(data_path):
    df = pd.read_csv(data_path)

    # Encoding name features with target encoding
    nameEncoder = TargetEncoder(cols=['name', 'sire', 'dam', 'bmSire'], smoothing=10.0)

    y = df['rating']

    names = df['name']

    X = df.drop(columns=['rating'])

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42)

    X_train = nameEncoder.fit_transform(X_train, y_train)  
    X_test = nameEncoder.transform(X_test)                 


    # rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=13)
    xgbRegressor = xgb.XGBRegressor(max_depth=6, 
                                    learning_rate=0.1, 
                                    n_estimators=100,  
                                    eval_metric=['logloss', 'auc'], 
                                    random_state=42)
  
    # Fitting the model
    xgbRegressor.fit(X_train, y_train, 
                     eval_set=[(X_train, y_train), (X_test, y_test)],  
                     verbose=False)


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

    # Display Predictions for first 10 samples
    #display_predictions(xgbRegressor, X_test, y_test, names_test,  num_predictions=10)

    # Display Training
    graph_training(xgbRegressor)


    
if __name__ == "__main__":
    data_path = "data/baseData.csv"
    train_model(data_path)