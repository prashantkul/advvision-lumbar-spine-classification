import pandas as pd
import tensorflow as tf


def augment(df, disease_predictions):
    return pd.merge(df, disease_predictions, on=['series_id'], how='left')


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

    # Extract features (X) and labels (y)
    X = df.drop(columns=['composite_key', 'series_id'] + label_columns).values  # Drop non-feature columns
    y = df[label_columns].applymap(lambda x: {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}[x]).values

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