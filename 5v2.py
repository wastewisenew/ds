import pandas as pd

df = pd.read_csv('as.csv').drop(columns=[
    'Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Former owner', 
    'Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']).set_index('Identifier')

df['Date of Publication'] = df['Date of Publication'].str[:4]
df['Place of Publication'] = df['Place of Publication'].replace({
    r'^\s*$': 'Unknown', '.*London.*': 'London', '.*Oxford.*': 'Oxford'
}, regex=True)

print(df.head())
