import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import RandomizedSearchCV , KFold
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from dataHandler import *
from evaluate import *


"""
Helper function to display the validation of the model during training.
"""
def graph_training(fold_results):
    """
    Args:
        fold_results: list of evals_result() dicts, one per fold
                      e.g. [{'validation_0': {'rmse': [...]}}, ...]
    """
    all_val_rmse = [fold['validation_0']['rmse'] for fold in fold_results]

    # Trim all curves to the shortest fold (early stopping may vary length)
    min_len = min(len(curve) for curve in all_val_rmse)
    all_val_rmse = np.array([curve[:min_len] for curve in all_val_rmse])

    mean_rmse = np.mean(all_val_rmse, axis=0)
    std_rmse  = np.std(all_val_rmse,  axis=0)
    x_axis    = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot each fold lightly in the background
    for i, curve in enumerate(all_val_rmse):
        ax.plot(x_axis, curve, color='steelblue', alpha=0.2, linewidth=1, label='Fold RMSE' if i == 0 else None)

    # Plot mean curve and std band
    ax.plot(x_axis, mean_rmse, color='steelblue', linewidth=2.5, label='Mean Val RMSE')
    ax.fill_between(x_axis, mean_rmse - std_rmse, mean_rmse + std_rmse,
                    alpha=0.2, color='steelblue', label='±1 Std Dev')

    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Validation RMSE Across K-Folds', fontsize=14, pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def train_model(data_path):

    # df = csv_to_dataframe(data_path)
    df = pd.read_csv(data_path, index_col=0)


    y = df['rating']
    names = df['name']
    X = df.drop(columns=['rating', 'name', 'sire', 'dam', 'bmSire', 'damForm', 'rawErg', 'avgBmSireForm'])  # Drop the target variable from the features and any other features for testing
    
    print("============ Sample training data ============")
    print(X.head(10))

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    #============== Hyperparameter Tuning with RandomizedSearchCV ==============

    # param_grid = {
    #     # 1. Tree structure — controls model complexity (REDUCED to prevent overfitting)
    #     "max_depth": [3, 4, 5, 6],                 # Shallower trees to reduce variance and spurious patterns
    #     "min_child_weight": [2, 4, 7, 10],         # Higher values = more conservative splits, reduces feature dominance

    #     # 2. Boosting strength — more trees with lower learning rate for better generalization
    #     "n_estimators": [200, 400, 600, 800],      # More trees with lower LR compensates
    #     "learning_rate": [0.005, 0.01, 0.03, 0.05],  # Lower LR = slower learning = better generalization (avoid 0.1)

    #     # 3. Sampling — AGGRESSIVE subsampling to increase diversity and reduce feature dominance
    #     "subsample": [0.6, 0.7, 0.8, 0.9],         # Row sampling - lower = more variance reduction
    #     "colsample_bytree": [0.5, 0.65, 0.8, 0.95],  # Feature sampling - CRITICAL: prevent same features dominating
    #     "colsample_bylevel": [0.5, 0.7, 0.9],      # Per-level feature subsampling for extra diversity

    #     # 4. Regularization — STRONG penalties to prevent overfitting and feature dominance
    #     "gamma": [0.1, 0.5, 1.0, 2.0],             # Higher = require more loss reduction to split (prevents overfitting)
    #     "reg_alpha": [0.01, 0.1, 0.5, 1.0],        # L1 regularization - forces feature selection diversity
    #     "reg_lambda": [2.0, 5.0, 10.0, 15.0],      # L2 regularization - reduces coefficient magnitude
    # }
    
    # xgbRegressor = xgb.XGBRegressor(random_state=42, 
    #                                 objective='reg:squarederror',
    #                                 n_jobs=-1,
    #                                 device='cuda',
    #                                 tree_method="hist",)
    # grid_search = RandomizedSearchCV(
    #     xgbRegressor,
    #     param_grid,
    #     cv=kfold,
    #     n_iter=100,
    #     scoring='neg_root_mean_squared_error',  # Use negative RMSE for regression
    #     n_jobs=-1,
    #     verbose=2
    # )


    # grid_search.fit(X, y)
    
    # print(f"Best parameters found: {grid_search.best_params_}")

    #============== Hyperparameter Tuning with RandomizedSearchCV ==============

    xgbRegressor = xgb.XGBRegressor(
        max_depth=5,             
        learning_rate=0.03,         
        n_estimators=400,            
        min_child_weight=10,          
        subsample=0.9,              
        colsample_bytree=0.9,        
        colsample_bylevel=0.9,       
        eval_metric='rmse',
        objective='reg:squarederror',
        early_stopping_rounds=40,
        reg_alpha=0.1,                
        reg_lambda=15.0,              
        gamma=0.1,
        random_state=42)

    # Out of fold predictions for evaluation and feature importance analysis
    predictions = np.zeros(len(y))
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgbRegressor.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],  # now early stopping works correctly
            verbose=False
        )

        predictions[val_idx] = xgbRegressor.predict(X_val)
        fold_results.append(xgbRegressor.evals_result())
    
    for name, importance in zip(X.columns, xgbRegressor.feature_importances_):
        print(name, importance)
    
    # Evaluating the model with predctions on X_test
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2  = r2_score(y, predictions)

    
    print("\n==== Model Evaluation Metrics: ====")
    print(f"MSE:  {mse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("===================================\n")

    # Display Predictions for first 10 samples
    display_predictions(predictions, X, y, names, num_predictions=15)

    # Display Training
    graph_training(fold_results)
    print(f"Feature Importances: {xgbRegressor.feature_importances_}")
    print(f"Best Iteration: {xgbRegressor.best_iteration}")

    # Display feature importance
    display_feature_importance(xgbRegressor, X)

    # Save the model to a file
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgbRegressor, f)

    
if __name__ == "__main__":
    data_path = "data/encodedHorseData.csv"
    train_model(data_path)