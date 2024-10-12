import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
df = pd.read_csv('movies.csv')  # Ensure you have the correct path to your CSV file

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("GPU is available and will be used.")
else:
    print("GPU is not available. Using CPU.")

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to encode text using BERT
def encode_text(text):
    if not isinstance(text, str):
        text = str(text)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token's embedding for simplicity (outputs[0][:, 0, :])
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Encode all movie scripts once
df['script_embedding'] = df['Script'].apply(encode_text)

# Function to perform the query
def perform_query(query, top_n):
    query_embedding = encode_text(query)
    df['similarity'] = df['script_embedding'].apply(lambda x: cosine_similarity(query_embedding, x).item())
    result_df = df.sort_values(by='similarity', ascending=False)
    top_movies = result_df[['ID', 'Title', 'similarity']].head(top_n)
    return top_movies
code2
# User interaction loop
while True:
    query = input("Enter your query (or type 'exit' to quit): ").strip().lower()
    if query == 'exit':
        break
    try:
        top_n = int(input("Enter the number of results you want to view: "))
        top_movies = perform_query(query, top_n)
        print("\nTop Movies for Query '{}':\n".format(query))
        print(top_movies)
        print("\n" + "-"*50 + "\n")
    except ValueError:
        print("Please enter a valid number for the number of results.")
