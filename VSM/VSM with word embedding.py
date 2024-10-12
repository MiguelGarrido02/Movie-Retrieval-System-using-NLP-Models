import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# Load the dataset
df = pd.read_csv('movies.csv')

# Check for missing values and handle them
df.dropna(subset=['Script'], inplace=True)
df['Script'] = df['Script'].astype(str)

# Tokenize the scripts
tokenized_scripts = [script.split() for script in df['Script']]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_scripts, vector_size=100, window=5, min_count=1, workers=4)

# Function to search movies using Word Embeddings
def search_movies_word2vec(query, num_results, word2vec_model, df):
    # Tokenize the query
    query_tokens = query.split()
    
    # Calculate the average word embedding for the query
    query_embedding = sum([word2vec_model.wv[token] for token in query_tokens if token in word2vec_model.wv]) / len(query_tokens)
    
    # Compute cosine similarity between query embedding and all movie scripts
    similarity_scores = []
    for script in tokenized_scripts:
        script_embedding = sum([word2vec_model.wv[word] for word in script if word in word2vec_model.wv]) / len(script)
        similarity_scores.append(cosine_similarity([query_embedding], [script_embedding])[0][0])
    
    # Get the top movie indices
    top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:num_results]
    
    # Retrieve the movie titles and similarities
    results = [(df.iloc[idx]['Title'], similarity_scores[idx]) for idx in top_indices]
    
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

    results = search_movies_word2vec(query, num_results, word2vec_model, df)

    # Print results
    print(f"\nTop {num_results} results for query '{query}':")
    for title, score in results:
        print(f"Title: {title}, Similarity: {score:.4f}")
    print("\n")  # Add some spacing between queries

