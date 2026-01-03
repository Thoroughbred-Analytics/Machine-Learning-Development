import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(data_path):
    df = pd.read_csv(data_path)

    y = df['rating']
    X = df.drop(columns=['rating'])

    X_rain, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)