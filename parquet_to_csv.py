import pandas as pd

df = pd.read_parquet("data/prompts_poc.parquet")

csv = df.to_csv('data/prompts_experiment_2.csv', index=False)