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
import matplotlib.pyplot as plt
import xgboost as xgb



def generate_tree(model, df, output_path='tree.png'):
    """
    Generates a visualization of the first tree in the RandomForest model.
    
    Args:
        model: Trained RandomForest model
        df: Pandas DataFrame with feature columns (used for feature names)
        output_path: Path to save the tree image
    """
    feature_names = df.columns.tolist()
    tree = model.estimators_[0]  # Visualize the first tree in the forest
    
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, max_depth=3)
    plt.savefig(output_path)
    print(f"Tree visualization saved to {output_path}")
    plt.close()



def train_model(data_path):
    df = pd.read_csv(data_path)

    # Encoding name features with target encoding
    nameEncoder = TargetEncoder(cols=['sire', 'dam', 'bmSire'], smoothing=10.0)

    y = df['rating']
    X = df.drop(columns=['rating', 'name'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = nameEncoder.fit_transform(X_train, y_train)  
    X_test = nameEncoder.transform(X_test)                 

    paramGrid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['squared_error', 'absolute_error', 'gini', 'entropy']
    }


    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=13)
    xgbRegressor = xgb.XGBRegressor()
    #gridSearch = GridSearchCV(estimator=rf, param_grid=paramGrid, cv=5, n_jobs=-1, verbose=2)

    # Fitting the model
    xgbRegressor.fit(X_train, y_train)

    # Generate tree visualization
    #generate_tree(rf, X_train)

    # Evaluating the model with predctions on X_test

    y_pred = xgbRegressor.predict(X_test)

    print(f"predictions: {y_pred[0:10]}\n")
    print(f"actual: {y_test.iloc[0:10].to_numpy()}\n")
    
    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)
    accuracy = xgbRegressor.score(X_test, y_test)
    
    # getting dummy regressor score for comparison
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_score = dummy.score(X_test, y_test)

    print(f"Dummy Regressor R2 Score: {dummy_score}")
    print(f"mse: {mse}")
    print(f"r2: {r2}")
    print(f"accuracy: {accuracy}")



if __name__ == "__main__":
    data_path = "data/baseData.csv"
    train_model(data_path)