
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('movies.csv')

# Check for missing values and handle them
df.dropna(subset=['Script'], inplace=True)
df['Script'] = df['Script'].astype(str)

# Initialize the CountVectorizer for Bag-of-Words
vectorizer = CountVectorizer()

# Fit and transform the scripts
bow_matrix = vectorizer.fit_transform(df['Script'])

# Function to search movies using Bag-of-Words
def search_movies_bow(query, num_results, bow_matrix, vectorizer, df):
    # Transform the query into a BoW vector
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between query vector and all movie scripts
    cosine_similarities = cosine_similarity(query_vec, bow_matrix).flatten()
    
    # Get the top movie indices
    top_indices = cosine_similarities.argsort()[-num_results:][::-1]  # Top 'num_results' results
    
    # Retrieve the movie titles and similarities
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

    results = search_movies_bow(query, num_results, bow_matrix, vectorizer, df)

    # Print results
    print(f"\nTop {num_results} results for query '{query}':")
    for title, score in results:
        print(f"Title: {title}, Similarity: {score:.4f}")
    print("\n")  # Add some spacing between queries
