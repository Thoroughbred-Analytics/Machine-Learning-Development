import pandas as pd
import time 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


def csv_to_dataframe(filePath: str):
    """
    Function to preprocess a csv file, returning a cleaned DataFrame.

    Input: path_to_csv: str - Path to the CSV file to be processed.

    Output: df: pd.DataFrame - The cleaned DataFrame.
        idToName: dict - Mapping from unique IDs back to horse names.
    """

    df = pd.read_csv(filePath, low_memory=False)

    df = df[df['name'] != 'Unnamed']

    print(f"Shape of dataset: {df.shape}")

    # Dropping columns we know we don't need:
    df = df.drop(columns=['ems', 'grade', 'grade4', 'code', 'lot', 'price', 'status', 'vendor', 'purchaser', 'prev. price', 'form'], axis=1)

    # converting fees to a numeric value
    df['fee'] = pd.to_numeric(df['fee'], errors='coerce')

    # removing horses with ratings of 0 -> means they haven't raced yet?
    df = df[df['rating'] > 0]


    # ---- Turning the birth year to the age of the horse ----
    year = time.localtime().tm_year
    df['yob'] = year - df['yob']
    df = df.rename(columns={'yob': 'age'})

    # ---- Encoding the ordinal features (form) ----
    ordinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value= -1)
    # encodedForm = ordinalEncoder.fit_transform(np.array(df['form']).reshape(-1,1))
    encodedFormDam = ordinalEncoder.fit_transform(np.array(df['form2']).reshape(-1,1))

    # saving the encoder in case we need it to reverse-engineer the encoder
    joblib.dump(ordinalEncoder, "data/ordinalEncoder.pk1")

    df = df.drop(['form2'], axis=1)
    # df['form'] = encodedForm
    df['damForm'] = encodedFormDam


    # ---- Encoding the names of the horses with label encoding ----
    uniqueNames = pd.concat([df['name'], df['sire'], df['dam'], df['bmSire']]).unique()
    nameToId = {name: idx for idx, name in enumerate(uniqueNames)}
    idToName = {idx: name for idx, name in enumerate(uniqueNames)}

    df['name_encoded'] = df['name'].map(nameToId)
    df['sire'] = df['sire'].map(nameToId)
    df['dam'] = df['dam'].map(nameToId)
    df['bmSire'] = df['bmSire'].map(nameToId)



    # ---- One hot encoding for gender ----
    hotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encodedSex = hotEncoder.fit_transform(df[['sex']])

    sexCols = hotEncoder.get_feature_names_out(['sex'])
    sexDf = pd.DataFrame(encodedSex, columns=sexCols, index=df.index)

    df = pd.concat([df.drop(columns=['sex']), sexDf], axis=1)

    print(f"number of unique names: {df['name'].nunique()}")
    print(f"number of unique sires: {df['sire'].max()}")
    print(f"number of unique dams: {df['dam'].max()}")
    print(f"number of unique bmSires: {df['bmSire'].max()}")


    # ---- Filling in missing values in Fee category ----
    df['fee'] = df['fee'].fillna(df['fee'].median())


    return df, idToName