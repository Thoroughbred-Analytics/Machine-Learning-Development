import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder

from category_encoders import TargetEncoder, BinaryEncoder
from dataHandler import *
from evaluate import *


"""
Helper function to display the validation of the model during training.
"""
def graph_training(model):
    results = model.evals_result()
    
    # DEBUGGING: Print the actual values
    print("DEBUG: Checking results dictionary structure")
    print("Keys:", results.keys())
    print("\nValidation_0 keys:", results['validation_0'].keys())
    print("Validation_1 keys:", results['validation_1'].keys())
    
    print("\nFirst 10 Train RMSE values:")
    print(results['validation_0']['rmse'][:10])
    
    print("\nFirst 10 Test RMSE values:")
    print(results['validation_1']['rmse'][:10])
    
    print("\nLast 10 Train RMSE values:")
    print(results['validation_0']['rmse'][-10:])
    
    print("\nLast 10 Test RMSE values:")
    print(results['validation_1']['rmse'][-10:])
    
    # Check if test values are actually changing
    test_rmse = results['validation_1']['rmse']
    print(f"\nTest RMSE - Min: {min(test_rmse):.6f}, Max: {max(test_rmse):.6f}")
    print(f"Test RMSE - Std Dev: {np.std(test_rmse):.6f}")
    
    # Original plotting code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # Plot 1: RMSE over time
    ax1.plot(x_axis, results['validation_0']['rmse'], label='Train', linewidth=2)
    ax1.plot(x_axis, results['validation_1']['rmse'], label='Test', linewidth=2)
    ax1.legend()
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Boosting Round')
    ax1.set_title('Training vs Test RMSE')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test RMSE only with zoomed scale
    ax2.plot(x_axis, results['validation_1']['rmse'], label='Test', linewidth=2, color='orange')
    ax2.legend()
    ax2.set_ylabel('RMSE')
    ax2.set_xlabel('Boosting Round')
    ax2.set_title('Test RMSE (Zoomed)')
    ax2.grid(True, alpha=0.3)

    # Let y-axis auto-scale to show any variation

    plt.tight_layout()
    plt.show()

def train_model(data_path):

    # df = csv_to_dataframe(data_path)
    df = pd.read_csv(data_path, index_col=0)

    # Testing out different encoders for the names
    # nameEncoder = TargetEncoder(cols=['name', 'sire', 'dam', 'bmSire'], smoothing=10.0)
    # binaryEncoder = BinaryEncoder(cols=['name', 'sire', 'dam', 'bmSire'], handle_unknown='ignore')


    y = df['rating']
    names = df['name']
    X = df.drop(columns=['rating', 'name', 'sire', 'dam', 'bmSire'])  # Drop the target variable from the features
    print(X.head())


    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42)

    # === For encoders besides the label encoder ===
    # X_train = nameEncoder.fit_transform(X_train, y_train)  
    # X_test = nameEncoder.transform(X_test)          
    
    """
    For grid searching
    param_grid = {
        'n_estimators': [10, 20, 30, 50],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9]
    }

    Best parameters found: {'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 50, 'subsample': 0.9}
    
    xgbRegressor = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        xgbRegressor,
        param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    """

    xgbRegressor = xgb.XGBRegressor(max_depth=8, 
                                    learning_rate=0.1, 
                                    n_estimators=50,  
                                    min_child_weight=3,
                                    subsample=0.8,
                                    eval_metric='rmse', 
                                    objective='reg:squarederror',
                                    early_stopping_rounds=10,             
                                    reg_alpha=0.1,          # L1 regularization
                                    reg_lambda=1.0,         # L2 regularization
                                    random_state=42)
  
    # Fitting the model
    xgbRegressor.fit(X_train, y_train, 
                     eval_set=[(X_train, y_train), (X_test, y_test)],  
                     verbose=False)


    for name, importance in zip(X_train.columns, xgbRegressor.feature_importances_):
        print(name, importance)

    # Evaluating the model with predctions on X_test
    y_pred = xgbRegressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n==== Model Evaluation Metrics: ====")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("===================================\n")

    # Display Predictions for first 10 samples
    display_predictions(xgbRegressor, X_test, y_test, names_test,  num_predictions=10, idToName=None)

    # Display Training
    graph_training(xgbRegressor)

    print(f"Feature Importances: {xgbRegressor.feature_importances_}")
    print(f"Best Iteration: {xgbRegressor.best_iteration}")

    # Save the model to a file
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgbRegressor, f)

    
if __name__ == "__main__":
    data_path = "data/encodedHorseData.csv"
    train_model(data_path)