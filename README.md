# ML-Text-Project-Sentiment-Analysis-for-IMDB-Review-Dataset

## Project Overview

In this project, we practice **Sentiment Analysis** using movie reviews from the **IMDB dataset**. Sentiment Analysis is a subfield of **Natural Language Processing (NLP)** that focuses on determining the **emotion or opinion expressed in a piece of text**.

The goal of this project is to classify movie reviews into **positive** or **negative** sentiments using **Machine Learning text classification techniques**.

Each text sample (review) will be assigned a label from a predefined set of categories.

Example:

| Review ID | Review                                        | Sentiment |
| --------- | --------------------------------------------- | --------- |
| 3537      | Quite what the producers of this appalling... | negative  |
| 3769      | My favourite police series of all time...     | positive  |

---

# Sentiment Analysis

Sentiment Analysis aims to evaluate customer opinions and classify them into categories such as:

* Positive
* Negative
* Neutral

In this project, we focus on **binary sentiment classification** using the **IMDB Movie Review Dataset**.

---

# Environment Setup

### Check Python Version

```bash
# Check the installed Python version
python -V
```

or

```bash
python --version
```

---

### Create Conda Environment

```bash
# Create a new conda environment for this project
conda create --name imdb_end python=3.11.7
```

---

### Activate Environment

```bash
# Activate the environment
conda activate imdb_end
```

---

# Dataset

Download the dataset:

```
IMDB-Dataset.csv
```

The dataset contains **50,000 movie reviews** labeled as **positive** or **negative**.

---

# Create Jupyter Notebook

Create a notebook file for experimentation:

```
analysisimdb.ipynb
```

---

# Install Required Libraries

### Install contractions library

```python
# Install contractions library used for expanding shortened words
!pip install -q contractions
```

Other required libraries:

```bash
pip install pandas nltk beautifulsoup4 seaborn matplotlib scikit-learn
```

---

# Dataset Loading

### Load dataset using pandas

```python
# Import pandas library
import pandas as pd

# Load the dataset
df = pd.read_csv('IMDB-Dataset.csv')

# Display first rows
df.head()

# Show dataset information
df.info()

# Statistical summary
df.describe()

# Access review column
df['review']

# Check missing values
df.isna().sum()
```

---

# Data Preprocessing

## Remove Duplicate Reviews

```python
# Select duplicate rows
duplicated_df = df[df.duplicated()]
duplicated_df

# Remove duplicates
df = df.drop_duplicates()

# Check dataset summary again
df.describe()
```

---

# Data Cleaning

We clean the dataset using the following steps:

* Remove HTML tags
* Remove punctuation
* Remove URLs
* Remove emojis
* Remove numbers
* Remove stopwords
* Lemmatization

Required libraries:

```python
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions
```

---

### Define Stop Words

```python
# Load English stopwords
stop = set(stopwords.words('english'))
```

---

### Expand Contractions

```python
# Convert words like "can't" to "cannot"
def expand_contractions(text):
    return contractions.fix(text)
```

---

### Text Preprocessing Function

```python
# Function to clean text data
def preprocess_text(text):

    wl = WordNetLemmatizer()

    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Expand contractions
    text = expand_contractions(text)

    # Remove emojis
    emoji_clean = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    text = emoji_clean.sub(r'',text)

    # Add space after periods
    text = re.sub(r'\.(?=\S)', '. ',text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation and convert to lowercase
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])

    # Lemmatization and stopword removal
    text = " ".join([
        wl.lemmatize(word)
        for word in text.split()
        if word not in stop and word.isalpha()
    ])

    return text
```

---

### Important Note about Stopwords

You can check stopwords using:

```python
stop
```

Example:

```
{'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', ...}
```

Check if a word is a stopword:

```python
"no" in stop
"not" in stop
```

**Important:**
Words like **"no"** or **"not"** are important for sentiment analysis because they indicate **negation**. Removing them may change the meaning of a sentence and affect model accuracy.

---

### Punctuation Reference

```python
string.punctuation
```

Output:

```
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
```

---

# Exploratory Data Analysis (EDA)

### Install visualization libraries

```bash
pip install seaborn matplotlib
```

---

# Sentiment Label Distribution

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

freq_pos = len(df[df['sentiment'] == 'positive'])
freq_neg = len(df[df['sentiment'] == 'negative'])

data = [freq_pos, freq_neg]
labels = ['positive', 'negative']

pie, ax = plt.subplots(figsize=[11,7])

plt.pie(
    x=data,
    autopct=lambda pct: func(pct, data),
    explode=[0.0025]*2,
    pctdistance=0.5,
    colors=[sns.color_palette()[0],'tab:red'],
    textprops={'fontsize': 16}
)

labels = ['Positive','Negative']
plt.legend(labels, loc="best", prop={'size': 14})

pie.savefig("PieChart.png")
plt.show()
```

---

# Review Length Analysis

```python
words_len = df['review'].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp['words length'] = words_len
```

---

# Positive Review Distribution

```python
hist_positive = sns.displot(
    data=df_temp[df_temp['sentiment']=='positive'],
    x="words length",
    hue="sentiment",
    kde=True,
    height=7,
    aspect=1.1
)
```

---

# Negative Review Distribution

```python
hist_negative = sns.displot(
    data=df_temp[df_temp['sentiment']=='negative'],
    x="words length",
    hue="sentiment",
    kde=True,
    palette=['red']
)
```

---

# Kernel Density Plot

```python
plt.figure(figsize=(7,7.1))

sns.kdeplot(
    data=df_temp,
    x="words length",
    hue="sentiment",
    fill=True
)
```

---

# Box Plot

```python
fig, ax = plt.subplots(figsize=(7,7))

sns.boxplot(
    hue="sentiment",
    y='words length',
    data=df_temp,
    palette=['green','red']
)
```

---

# Most Common Words

```python
from collections import Counter

words = ' '.join(df['review']).split()
counter = Counter(words)

most = counter.most_common()

vocabs, word_counts = [], []

for word, count in most:
    if word not in stop:
        vocabs.append(word)
        word_counts.append(count)
```

---

# Word Frequency Visualization

```python
plt.figure(figsize=(9,7))

sns.barplot(x=word_counts[:10],y=vocabs[:10])

plt.title("Most Common Words in Reviews")
```

---

# Text Encoding

To train machine learning models, we convert text into numerical vectors.

---

### Import ML libraries

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
```

---

### Encode Labels

```python
label_encode = LabelEncoder()

y_data = label_encode.fit_transform(df['sentiment'])
```

Mapping:

```
0 → negative
1 → positive
```

---

### Split Dataset

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=42
)
```

---

### TF-IDF Vectorization

```python
tfidf_vectorizer = TfidfVectorizer(max_features=10000)

tfidf_vectorizer.fit(x_train, y_train)

x_train_encoded = tfidf_vectorizer.transform(x_train)
x_test_encoded = tfidf_vectorizer.transform(x_test)
```

---

# Model Training

Example prediction using a classifier:

```python
df['review'][:2]

example_encoded = tfidf_vectorizer.transform(df['review'][:2])

example_pred = dt_classifier.predict(example_encoded)

label_encode.inverse_transform(example_pred)
```

---

# Project Structure

```
imdb_sentiment_analysis
│
├── analysisimdb.ipynb
├── IMDB-Dataset.csv
└── README.md
```

---

# Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* TF-IDF
* Seaborn
* Matplotlib
* BeautifulSoup

---
