import tensorflow as tf

class TFSampleIDGenerator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_count = 0

    def generate_ids(self, num_samples=None):
        """
        Generator function to yield sample IDs for a specific number of samples.
        
        :param num_samples: Number of samples to generate IDs for. If None, generates for all samples.
        """
        dataset = self.dataset.take(num_samples) if num_samples is not None else self.dataset
        
        for batch in dataset:
            if isinstance(batch, tuple):
                batch_size = batch[0].shape[0]  # Assuming the first element is the data
            else:
                batch_size = 1  # Single element, not a batch
            
            for _ in range(batch_size):
                self.sample_count += 1
                yield f"Sample_{self.sample_count}"

    def get_ids(self, num_samples=None):
        """
        Returns a list of sample IDs for a specific number of samples.
        
        :param num_samples: Number of samples to generate IDs for. If None, generates for all samples.
        """
        return list(self.generate_ids(num_samples))

# Usage example:
# dataset = ...  # Your TF dataset
# id_generator = TFSampleIDGenerator(dataset)
# sample_ids = id_generator.get_ids(4)  # Generate IDs for exactly 4 samples