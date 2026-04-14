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

def find_best_hyperparameters(data_path):
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
    
    #  Hyperparameter Tuning with RandomizedSearchCV
    #  
    param_grid = {
        # 1. Tree structure — controls model complexity (REDUCED to prevent overfitting)
        "max_depth": [3, 4, 5, 6],                 # Shallower trees to reduce variance and spurious patterns
        "min_child_weight": [2, 4, 7, 10],         # Higher values = more conservative splits, reduces feature dominance

        # 2. Boosting strength — more trees with lower learning rate for better generalization
        "n_estimators": [200, 400, 600],      # More trees with lower LR compensates
        "learning_rate": [0.005, 0.01, 0.03, 0.05],  # Lower LR = slower learning = better generalization (avoid 0.1)

        # 3. Sampling — AGGRESSIVE subsampling to increase diversity and reduce feature dominance
        "subsample": [0.6, 0.7, 0.8, 0.9],         # Row sampling - lower = more variance reduction
        "colsample_bytree": [0.5, 0.65, 0.8, 0.95],  # Feature sampling - CRITICAL: prevent same features dominating
        "colsample_bylevel": [0.5, 0.7, 0.9],      # Per-level feature subsampling for extra diversity

        # 4. Regularization — STRONG penalties to prevent overfitting and feature dominance
        "gamma": [0.1, 0.5, 1.0, 2.0],             # Higher = require more loss reduction to split (prevents overfitting)
        "reg_alpha": [0.01, 0.1, 0.5, 1.0],        # L1 regularization - forces feature selection diversity
        "reg_lambda": [2.0, 5.0, 10.0, 15.0],      # L2 regularization - reduces coefficient magnitude

        "colsample_bynode": [0.5, 0.7, 0.9],       # Per-split feature sampling (finer than bylevel)
        "max_delta_step": [0, 1, 5],               # Useful for imbalanced targets / capping updates
        "max_leaves": [0, 16, 32, 64],             # Only relevant when grow_policy='lossguide'
        "rate_drop": [0.1, 0.2],                   # For DART booster only
        "booster": ["gbtree", "dart"],             # DART adds dropout regularization
    }

    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    xgbRegressor = xgb.XGBRegressor(random_state=42, 
                                    objective='reg:squarederror',
                                    device='cuda',
                                    tree_method="hist",)
    grid_search = RandomizedSearchCV(
        xgbRegressor,
        param_grid,
        scoring=scoring,
        refit='rmse',   # Still optimize for RMSE, but log others
        cv=kfold,
        n_iter=300,
        n_jobs=1,
        verbose=2
    )


    grid_search.fit(X, y)
    
    print(f"Best parameters found: {grid_search.best_params_}")


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
    

    xgbRegressor = xgb.XGBRegressor(
        max_depth=6,             
        learning_rate=0.03,         
        n_estimators=200,            
        min_child_weight=7,          
        subsample=0.9,              
        colsample_bytree=0.95,        
        colsample_bylevel=0.9,
        colsample_bynode=0.7,       
        eval_metric='rmse',
        objective='reg:squarederror',
        early_stopping_rounds=40,
        reg_alpha=1.0,                
        reg_lambda=2.0,              
        gamma=2.0,
        rate_drop=0.2,
        max_leaves=64,
        max_delta_step=0,
        booster='gbtree',
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
    #find_best_hyperparameters(data_path)