import pandas as pd
import numpy as np
books_df = pd.read_csv('desktop/BL-Flickr-Images-Book.csv')
print("Original DataFrame:")
print(books_df.head())
columns_to_drop = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Formerowner', 'Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']
books_df.drop(columns=columns_to_drop, inplace=True)
books_df.set_index('Identifier', inplace=True)
def clean_date(date):
    if isinstance(date, str):
        match = re.search(r'\d{4}', date)
    if match:
        return match.group()
    return np.nan
books_df['Date of Publication'] = books_df['Date of Publication'].apply(clean_date)
books_df['Place of Publication'] = np.where(
books_df['Place of Publication'].str.contains('London'),
'London',
np.where(
books_df['Place of Publication'].str.contains('Oxford'),
'Oxford',
books_df['Place of Publication'].replace(
r'^\s*$', 'Unknown', regex=True
)
)
)