import pandas as pd
import os

# Paths to dataset files
politifact_real_path = 'Dataset/politifact_real.csv'
politifact_fake_path = 'Dataset/politifact_fake.csv'
gossipcop_real_path = 'Dataset/gossipcop_real.csv'
gossipcop_fake_path = 'Dataset/gossipcop_fake.csv'

# Function to load and process a dataset
def load_dataset(path, label, source):
    try:
        df = pd.read_csv(path)
        # Add a label column (1 for real, 0 for fake)
        df['label'] = label
        # Add source information
        df['source'] = source
        print(f"Loaded {path}: {df.shape[0]} rows, columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# Load all datasets
politifact_real = load_dataset(politifact_real_path, 1, 'politifact')
politifact_fake = load_dataset(politifact_fake_path, 0, 'politifact')
gossipcop_real = load_dataset(gossipcop_real_path, 1, 'gossipcop')
gossipcop_fake = load_dataset(gossipcop_fake_path, 0, 'gossipcop')

# Check if all datasets were loaded correctly
datasets = [politifact_real, politifact_fake, gossipcop_real, gossipcop_fake]
if None in datasets:
    print("Failed to load one or more datasets. Exiting.")
    exit(1)

# Combine all datasets
combined_df = pd.concat(datasets, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")

# Save combined dataset
output_path = 'Dataset/combined_fake_news.csv'
combined_df.to_csv(output_path, index=False)
print(f"Combined dataset saved to {output_path}")

# Display dataset information
print("\nDataset Information:")
print(f"Total samples: {combined_df.shape[0]}")
print(f"Real news: {combined_df[combined_df['label'] == 1].shape[0]}")
print(f"Fake news: {combined_df[combined_df['label'] == 0].shape[0]}")
print(f"Sources: {', '.join(combined_df['source'].unique())}") 