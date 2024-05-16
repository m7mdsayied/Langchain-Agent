#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the Excel file skipping the initial rows with the message
df = pd.read_excel('NileULex_v0.27.xlsx', skiprows=10, names=['Term', 'Polarity', 'Egyptian', 'MSA'])

# Check the first few rows of the DataFrame to verify column names and data
print(df.head())


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.dropna(inplace=True)


# In[6]:


# Specify the number of random samples you want to choose
def generate_random_sample(df, n_samples=15):
    return df.sample(n=n_samples)
random_samples = generate_random_sample(df)
print(random_samples)


# # Calculate the total number of rows in the dataset

# In[4]:


total_rows = len(df)
# Count the occurrences of each type of polarity in the Polarity column
polarity_counts = df['Polarity'].value_counts()
descriptive_stats = df.describe()
print(total_rows)
print(polarity_counts)
print(descriptive_stats)


# # Perform Exploratory Data Analysis (EDA)
# # Create visualizations to explore the distribution of the Polarity column

# In[12]:


import matplotlib.pyplot as plt

# Bar plot for Polarity counts
plt.figure(figsize=(8, 6))
polarity_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Polarity in the Dataset')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Calculate the percentage of each type of polarity in the dataset
polarity_percentages = (polarity_counts / total_rows) * 100

# Print the dataset description, number of rows, and polarity counts
print("Dataset Description:")
print(descriptive_stats)
print("\nNumber of Rows in the Dataset:", total_rows)
print("\nNumber of Positive, Negative, Compound_Pos, and Compound_Neg in the Polarity column:")
print(polarity_counts)
print("\nPercentage of Each Polarity:")
print(polarity_percentages)


# # Model Loading and Training

# Choose an open source LLM

# Trial1: 
# Mistral-7B-v0.1 - You must be authenticated to access it.

# In[ ]:


from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the Arabic-specific model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[8]:


random_samples = generate_random_sample(df)
print(random_samples)


# # Command R+

# In[9]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-plus")
model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-plus")


# In[ ]:




