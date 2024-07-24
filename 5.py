import pandas as pd
import numpy as np
import re

# Load the data
books_df = pd.read_csv('as.csv')

# Display the original DataFrame
print("Original DataFrame:")
print(books_df.head())

# Drop irrelevant columns
columns_to_drop = ['Edition Statement', 'Corporate Author', 'Corporate Contributors', 'Formerowner', 'Engraver', 'Contributors', 'Issuance type', 'Shelfmarks']
books_df.drop(columns=columns_to_drop, inplace=True)

# Set index
books_df.set_index('Identifier', inplace=True)

# Clean the date of publication
def clean_date(date):
    if isinstance(date, str):
        match = re.search(r'\d{4}', date)
        if match:
            return match.group()
    return np.nan

books_df['Date of Publication'] = books_df['Date of Publication'].apply(clean_date)

# Clean the place of publication
books_df['Place of Publication'] = np.where(
    books_df['Place of Publication'].str.contains('London', na=False),
    'London',
    np.where(
        books_df['Place of Publication'].str.contains('Oxford', na=False),
        'Oxford',
        books_df['Place of Publication'].replace(r'^\s*$', 'Unknown', regex=True)
    )
)

# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(books_df.head())
