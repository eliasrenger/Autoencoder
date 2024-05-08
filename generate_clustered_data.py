# external imports
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# internal imports
from config import *

def generate_dataset(num_clusters, num_samples_per_cluster, cluster_variance, num_epochs, num_batches):
    data = []
    saved_centers = []
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # Generate random cluster centers
            centers = np.random.rand(num_clusters, 2) * 0.1  # Adjust 100 based on your data range
            saved_centers.append(centers)
            # Generate synthetic data with specified cluster variance
            X, _ = make_blobs(n_samples=num_samples_per_cluster*num_clusters, centers=centers, cluster_std=cluster_variance)
            
            # Add identifiers for trial and generation
            trial_id = [epoch] * (num_samples_per_cluster * num_clusters)
            generation_id = [batch] * (num_samples_per_cluster * num_clusters)
            
            # Combine data points with identifiers
            trial_data = np.column_stack((X, trial_id, generation_id))
            
            # Append trial data to main dataset
            data.extend(trial_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['x', 'y', 'epoch', 'batch'])
    
    return df, saved_centers

# Example usage:
num_clusters = 3
num_samples_per_cluster = 10
cluster_variance = 0.0001
num_epochs = 5
num_batches = 10

df, centers = generate_dataset(num_clusters, num_samples_per_cluster, cluster_variance, num_epochs, num_batches)
df.to_csv(f'data/clustered_data.csv', index=True)
# Display the generated DataFrame
print(centers)
