import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")

print(os.listdir(path))

file_path = os.path.join(path, "results.csv")
df = pd.read_csv(file_path)

df.to_csv('data/international_results.csv', index=False)