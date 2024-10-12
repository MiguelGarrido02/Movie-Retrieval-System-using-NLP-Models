import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load your dataset
df = pd.read_csv('movies.csv')

# Handle missing values in the 'Script' column
df['Script'] = df['Script'].fillna('')

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess the text by tokenizing, converting to lowercase, and removing stopwords."""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

# Preprocess the scripts
df['tokenized_script'] = df['Script'].apply(preprocess_text)

# Initialize BM25
corpus = df['tokenized_script'].tolist()
bm25 = BM25Okapi(corpus)

def search_movies(query, top_n, bm25, df):
    """
    Function to search for movies based on a query and return top N results.

    Parameters:
    query (str): The search query.
    top_n (int): Number of top results to return.
    bm25 (BM25Okapi): The BM25 model.
    df (pd.DataFrame): The DataFrame containing movie scripts.

    Returns:
    pd.DataFrame: DataFrame containing the top N movies with their BM25 scores.
    """
    # Process the query
    tokenized_query = preprocess_text(query)

    # Get BM25 scores for the query
    scores = bm25.get_scores(tokenized_query)

    # Add scores to DataFrame and sort
    df['bm25_score'] = scores
    df_sorted = df.sort_values(by='bm25_score', ascending=False)

    # Retrieve top N results
    top_movies = df_sorted.head(top_n)

    return top_movies[['ID', 'Title', 'bm25_score']]

# Interactive loop for user to make queries
while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    
    try:
        top_n_results = int(input("Enter the number of top results you want: "))
    except ValueError:
        print("Please enter a valid number.")
        continue

    if not user_query.strip():
        print("Query cannot be empty. Please try again.")
        continue

    try:
        top_movies = search_movies(user_query, top_n_results, bm25, df)
        print(top_movies)
    except Exception as e:
        logging.error("An error occurred while searching for movies: %s", str(e))
        print("An error occurred. Please try again.")

