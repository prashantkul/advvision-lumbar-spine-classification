import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def augment(df, disease_predictions, severity):
    """
    Augment the DataFrame with disease predictions and severity predictions.

    Parameters:
    - df: Original DataFrame.
    - disease_predictions: DataFrame containing disease predictions.
    - severity_df: DataFrame containing severity predictions (train.csv).

    Returns:
    - Merged DataFrame with added predictions.
    """
    df['series_id'] = df['series_id'].astype('int64')
    df['study_id'] = df['study_id'].astype('int64')
    disease_predictions['series_id'] = disease_predictions['series_id'].astype('int64')
    severity['study_id'] = severity['study_id'].astype('int64')

    # Merge the disease predictions
    merged_df = pd.merge(df, disease_predictions, on='series_id', how='left')

    # Merge the severity predictions from train.csv

    # Determine which column to use for the join
    join_column = 'study_id' if 'study_id' in merged_df.columns else 'study_id_x'

    # Perform the merge using the determined column
    merged_df = pd.merge(merged_df, severity, left_on=join_column, right_on='study_id', how='left')

    merged_df.columns = [col[:-2] if col.endswith(('_x', '_y')) else col for col in merged_df.columns]
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Label encode severity levels
    label_encoders = {}
    for column in severity.columns:
        if column not in ['study_id', 'study_id_x'] and column in merged_df.columns:
            if severity[column].dtype == 'object':
                le = LabelEncoder()
                merged_df[column] = le.fit_transform(merged_df[column])
                label_encoders[column] = le  # Save the encoder for potential inverse transformation
        else:
            print(f"Skipping {column}: not found in merged_df")

    return merged_df


def decode_severity_labels(encoded_df):
    """
    Decode the numeric severity levels in a DataFrame back to their original string labels.

    Parameters:
    - encoded_df: The DataFrame containing the encoded severity levels.

    Returns:
    - decoded_df: A DataFrame with the decoded severity levels.
    """
    decoded_df = encoded_df.copy()

    # Initialize a LabelEncoder instance for each column to decode
    label_encoders = {}

    for column in decoded_df.select_dtypes(include=['int64', 'int32']).columns:     
            # Create and fit a LabelEncoder based on the unique values in the column
            le = LabelEncoder()
            le.fit(['Normal/Mild', 'Moderate', 'Severe'])  # Assuming these are the possible severity levels
            
            # Decode the column using the fitted LabelEncoder
            decoded_df[column] = le.inverse_transform(decoded_df[column])
            label_encoders[column] = le  # Store the encoder if needed later

    return decoded_df

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


def tensorize(df, disease_predictions, label_columns, severity, padding_size=None):
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
    df = augment(df, disease_predictions, severity)
    df, _ = label_encode_strings(df)

    # Extract features (X) and labels (y)
    X = df.drop(columns=['composite_key', 'series_id', 'condition'] + list(severity.columns), errors='ignore').values
    y = df[severity.columns].values

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