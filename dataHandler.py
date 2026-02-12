import pandas as pd
import numpy as np
import time 
import joblib


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

"""
    Maps horse form strings to numerical values based on hierarchy.
    For compound strings, takes the highest value substring.
    
    Parameters:
    -----------
    column : pd.Series
        Column containing horse form strings
        
    Returns:
    --------
    pd.Series
        Column with mapped numerical values (0-4)
"""

def mapFormToHierarchy(column):
    formHierarchy = {
        'G1w': 4,      
        'G2w': 4,      
        'G3w': 3,      
        'G1p': 3,      
        'G2p': 2,      
        'G3p': 2,      
        'BTw': 2,      
        'W': 2,        
        'P': 1,        
        'UR': 0,       
        'UP': 0,       
        'UNKNOWN': 0,  
        np.nan: 0,     
        '': 0          
    }
    
    def getMaxValue(formString):
        # Handle NaN and empty strings
        if pd.isna(formString) or formString == '':
            return 0
        
        # Check if exact match exists
        if formString in formHierarchy:
            return formHierarchy[formString]
        
        # For compound strings, find the highest value substring
        maxValue = 0
        for key, value in formHierarchy.items():
            if key and not pd.isna(key) and key != '' and key in str(formString):
                maxValue = max(maxValue, value)
        
        return maxValue
    
    return column.apply(getMaxValue)

"""
Function to preprocess a csv file, returning a cleaned DataFrame.

Input: path_to_csv: str - Path to the CSV file to be processed.

Output: df: pd.DataFrame - The cleaned DataFrame.
        idToName: dict - Mapping from unique IDs back to horse names.
"""
def csv_to_dataframe(path_to_csv):

    df = pd.read_csv(path_to_csv, low_memory=False)

    df = df[df['name'] != 'Unnamed']
    print(f"Shape of dataset: {df.shape}")

    # Dropping columns we know we don't need:
    df = df.drop(columns=['ems', 'grade', 'grade4', 'code', 'lot', 'price', 'status', 'vendor', 'purchaser', 'prev. price'])

    # converting fees to a numeric value
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce')

    # removing horses with ratings of 0 -> means they haven't raced yet?
    df = df[df['rating'] > 0]

    # ---- Turning the birth year to the age of the horse ----
    df['yob'] = 2026 - df['yob']
    df = df.rename(columns={'yob': 'age'})
    
    # Encode both form columns
    df['form'] = df['form'].fillna('UNKNOWN')
    df['form2'] = df['form2'].fillna('UNKNOWN')
    
    df['form'] = mapFormToHierarchy(df['form']).astype(int)
    df['damForm'] = mapFormToHierarchy(df['form2']).astype(int)

    # Drop original form2 column
    df = df.drop(['form2'], axis=1)
    
    # ---- Encoding the names of the horses with label encoding ----
    uniqueNames = pd.concat([df['name'], df['sire'], df['dam'], df['bmSire']]).unique()
    nameToId = {name: idx for idx, name in enumerate(uniqueNames)}

    df['name'] = df['name'].map(nameToId)
    df['sire'] = df['sire'].map(nameToId)
    df['dam'] = df['dam'].map(nameToId)
    df['bmSire'] = df['bmSire'].map(nameToId)

    # ---- One hot encoding for gender ----
    hotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encodedSex = hotEncoder.fit_transform(df[['sex']])
    sexCols = hotEncoder.get_feature_names_out(['sex'])
    sexDf = pd.DataFrame(encodedSex, columns=sexCols, index=df.index)
    df = pd.concat([df.drop(columns=['sex']), sexDf], axis=1)
    
    # ---- Filling in missing values in Fee category ----
    df['fee'] = df['fee'].fillna(df['fee'].median())
    

    return df