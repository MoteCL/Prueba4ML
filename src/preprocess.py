import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import string

nltk.download('stopwords')
nltk.download('punkt')

# Load raw data
df = pd.read_csv('data/raw/spam.csv', encoding='latin1')
df.rename(columns={'Category': 'target', 'Message': 'text'}, inplace=True)

# Handle missing values
df['text'] = df['text'].fillna('')

# Encode target labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Preprocess text
ps = nltk.stem.porter.PorterStemmer()

def transform_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = [i for i in text if i.isalnum()]
        y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
        y = [ps.stem(i) for i in y]
        return " ".join(y)
    return ""

df['transformed_text'] = df['text'].apply(transform_text)

# Save preprocessed data
df.to_csv('data/preprocessed/preprocessed_data.csv', index=False)
