import pandas as pd



df = pd.read_csv('./test_df.csv', index_col = 0)
print(df.head())
