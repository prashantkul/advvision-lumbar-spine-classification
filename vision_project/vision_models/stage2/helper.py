import pandas as pd
import tensorflow as tf

def tensorize(dataset):
    predictions_df = pd.get_dummies(dataset, columns=predictions_df.columns.difference(['study_id']))

    # Extract the study_id as a separate array
    study_ids = predictions_df['study_id'].values
    features = predictions_df.drop(columns=['study_id']).values
    # Create a dictionary to store tensors for each study_id
    tensors_dict = {study_id: tf.convert_to_tensor(features[i], dtype=tf.float32) 
                    for i, study_id in enumerate(study_ids)}

    # Calculate the maximum length (number of features) among all tensors
    max_length = max(tensor.shape[0] for tensor in tensors_dict.values())

    padded_tensors_dict = {}
    for study_id, tensor in tensors_dict.items():
        # Calculate the padding needed
        padding_size = max_length - tensor.shape[0]
        
        # Pad the tensor on the right side (after the tensor's last element)
        # If padding_size > 0, pad with zeros
        padded_tensor = tf.pad(tensor, paddings=[[0, padding_size]])
        
        # Add the padded tensor to the new dictionary
        padded_tensors_dict[study_id] = padded_tensor

    return padded_tensors_dict
