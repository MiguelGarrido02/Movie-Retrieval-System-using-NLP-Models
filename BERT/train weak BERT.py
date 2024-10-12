import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import h5py

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Training script: GPU is available and will be used.")
else:
    print("Training script: GPU is not available. Using CPU.")

# Load the CSV file
df = pd.read_csv('movies.csv')  # Ensure you have the correct path to your CSV file

# Drop rows with any missing values
df = df.dropna(subset=['ID', 'Title', 'Script'])

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to encode text using BERT for batch processing
def encode_texts(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token's embedding for simplicity (outputs[0][:, 0, :])
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Encode all movie scripts in batches and store embeddings
batch_size = 8  # You can adjust the batch size based on your GPU memory capacity
all_embeddings = []
for i in range(0, len(df), batch_size):
    batch_texts = df['Script'][i:i+batch_size].tolist()
    batch_embeddings = encode_texts(batch_texts)
    all_embeddings.append(batch_embeddings)

# Concatenate all batch embeddings into a single NumPy array
all_embeddings = np.concatenate(all_embeddings, axis=0)

# Save the embeddings to an HDF5 file for efficient storage and retrieval
with h5py.File('script_embeddings.h5', 'w') as f:
    f.create_dataset('embeddings', data=all_embeddings)
    f.create_dataset('ids', data=df['ID'].values.astype('S'))
    f.create_dataset('titles', data=df['Title'].values.astype('S'))

# Save the model to disk
torch.save(model.state_dict(), 'bert_model_weak.pth')
