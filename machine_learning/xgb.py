import pandas as pd
import numpy as np
import seaborn as sns

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




def display_predictions(model, X_test, y_test, names_test, num_predictions=10):
    """
    Displays a comparison of predicted and actual values for the test set.
    
    Args:
        model: Trained model
        X_test: Test feature set
        y_test: Actual target values for the test set
        names_test: Names corresponding to the test set
        num_predictions: Number of predictions to display
    """
    y_pred = model.predict(X_test)
    print("Predicted vs Actual values:")
    for i in range(num_predictions):
        print(f"{names_test.iloc[i]} -  Predicted: {y_pred[i]:.2f}, Actual: {y_test.iloc[i]}")
    
    # Create a dataframe for plotting
    plot_data = pd.DataFrame({
        'Name': list(names_test.iloc[:num_predictions]) * 2,
        'Value': list(y_pred[:num_predictions]) + list(y_test.iloc[:num_predictions]),
        'Type': ['Predicted'] * num_predictions + ['Actual'] * num_predictions
    })
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Plot the connected dots
    for i in range(num_predictions):
        plt.plot([i, i], 
                [y_pred[i], y_test.iloc[i]], 
                'gray', 
                linestyle='--', 
                linewidth=1, 
                alpha=0.5)
    
    # Plot the points
    sns.scatterplot(data=plot_data, x='Name', y='Value', hue='Type', 
                   style='Type', s=100, palette=['#ff7f0e', '#1f77b4'])
    
    plt.xlabel('Horse', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.title('Predicted vs Actual Ratings', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Type', fontsize=10)
    plt.tight_layout()
    plt.show()




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