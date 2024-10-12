import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('movies.csv')

# Check for missing values and handle them
df.dropna(subset=['Script'], inplace=True)

# Preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords, punctuation, and special characters
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation and not token.isdigit()]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# BoW Vectorizer with bi-grams and tri-grams
bow_vectorizer = CountVectorizer(analyzer='word', tokenizer=preprocess_text, ngram_range=(1, 3))

# Train BoW model
bow_matrix = bow_vectorizer.fit_transform(df['Script'])

# Function to search movies using BoW
def search_movies_bow(query, num_results):
    query_vec = bow_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, bow_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-num_results:][::-1]
    results = [(df.iloc[idx]['Title'], cosine_similarities[idx]) for idx in top_indices]
    return results

# Main loop for user input
while True:
    query = input("Enter your query (or type 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        break
    
    try:
        num_results = int(input("Enter the number of results you want: ").strip())
    except ValueError:
        print("Invalid input. Please enter an integer for the number of results.")
        continue

    # Call the search_movies_bow function with query and num_results
    results = search_movies_bow(query, num_results)

    # Print the results
    print(f"\nTop {num_results} results using BoW for query '{query}':")
    for title, score in results:
        print(f"Title: {title}, Similarity: {score:.4f}")
    print("\n")
