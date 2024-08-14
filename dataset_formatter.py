import pandas as pd

# Load the current dataset
df = pd.read_csv('data/spam_ham_dataset.csv')

# Select and rename the relevant columns
df_reformatted = df[['text', 'label']]
df_reformatted.columns = ['email', 'label']

# Save the new dataset to a new CSV file
df_reformatted.to_csv('data/emails_reformatted.csv', index=False)
