import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def augment(df, disease_predictions):
    df['series_id'] = df['series_id'].astype('int64')
    disease_predictions['series_id'] = disease_predictions['series_id'].astype('int64')
    return pd.merge(df, disease_predictions, on='series_id', how='left')

def label_encode_strings(df):
    """
    Label encodes all string (object) columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing string columns to encode.

    Returns:
    pd.DataFrame: DataFrame with string columns label encoded.
    dict: A dictionary of LabelEncoders used for each column.
    """
    label_encoders = {}
    df_encoded = df.copy()  # Make a copy of the DataFrame to avoid modifying the original

    for column in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le  # Save the encoder for potential inverse transformation

    return df_encoded, label_encoders


def tensorize(df, disease_predictions, label_columns, padding_size=None):
    """
    Convert a DataFrame into a TensorFlow Dataset with padded tensors for multi-label classification.

    Parameters:
    - df: pandas DataFrame containing the data to be tensorized.
    - disease_predictions: Additional data or parameters required by the augment function.
    - label_columns: A list of column names representing the labels (severity levels).
    - padding_size: Optional; the size to pad the tensors to. If None, padding will be based on the largest item.

    Returns:
    - dataset: A TensorFlow Dataset containing the padded features and labels.
    """
    # Augment the dataframe
    df = augment(df, disease_predictions)
    df, _ = label_encode_strings(df)

    # Extract features (X) and labels (y)
    X = df.drop(columns=['composite_key', 'series_id', 'condition']).values
    y = df[label_columns]

    # Convert to TensorFlow tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    # Determine padding size
    if padding_size is None:
        max_size = tf.reduce_max([tf.shape(X_tensor)[0]])
    else:
        max_size = padding_size

    # Pad the tensors to the max size
    X_padded = tf.pad(X_tensor, paddings=[[0, max_size - tf.shape(X_tensor)[0]], [0, 0]], mode='CONSTANT')
    y_padded = tf.pad(y_tensor, paddings=[[0, max_size - tf.shape(y_tensor)[0]], [0, 0]], mode='CONSTANT')

    # Create a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_padded, y_padded))

    return dataset