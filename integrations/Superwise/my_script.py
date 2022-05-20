
import argparse
import joblib
import os
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


feature_column_names = ['longitude',
 'latitude',
 'housing_median_age',
 'total_rooms',
 'total_bedrooms',
 'population',
 'households',
 'median_income']

label_column = 'median_house_value'

id_column = 'id_col'


# inference functions
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(input_data, content_type):
    """Parse input data payload

    We expect a serialized numpy array as input.
    The numpy array should contain, as the first column, a unique ID, followed by the feature values.
    """
    if content_type == "application/x-npy":
        # Read the raw input
        load_bytes = BytesIO(input_data)
        input_np = np.load(load_bytes, allow_pickle=True)
        num_columns = input_np.shape[1]
        
        # Expect the first column to be a unique ID 
        if num_columns != len(feature_column_names) + 1: 
            raise ValueError(f"payload needs to contain {len(feature_column_names) + 1} columns, with first column being the ID of the record")
        df_columns = [id_column] + feature_column_names
        df = pd.DataFrame(data=input_np, columns=df_columns) 
        return df
    else:
        raise ValueError(f"content type {content_type} is not supported by this inference endpoint. Please send a legal application/x-npy payload")

        
# TODO - check if need to roll it back to the client or will SM track it as part of the input-output correlation
                         
def predict_fn(input_data, model):
    # predict using the features
    prediction = model.predict(input_data[feature_column_names])
    # Concatenate the ID column to the predictions to keep track 
    return np.stack((input_data[id_column], prediction), axis=1 )


