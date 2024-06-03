import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load and preprocess the data
df = pd.read_csv('/Users/trungdungle/Desktop/các hệ hỗ trợ ra quyết định /final project /code chính /Model/processed_laptop_data_final2.csv')

# Create the TF-IDF vectorizer and matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the DataFrame, TF-IDF vectorizer, TF-IDF matrix, and cosine similarity matrix
with open('model/laptop_recommender.pkl', 'wb') as f:
    pickle.dump((df, tfidf, tfidf_matrix, cosine_sim), f)

print("Model and preprocessors saved successfully.")
