import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
import re
import networkx as nx
from tqdm import tqdm
import os
import time

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Apply final linear layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def create_graph_from_text(text, tfidf_matrix, tfidf_feature_names, threshold=0.3):
    # Create a graph
    G = nx.Graph()
    
    # Extract words from text
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Add nodes
    for i, word in enumerate(words):
        if word in tfidf_feature_names:
            feature_idx = tfidf_feature_names.index(word)
            G.add_node(i, word=word, feature_idx=feature_idx)
    
    # Add edges between words that co-occur within a window
    window_size = 5
    for i in range(len(words)):
        for j in range(i+1, min(i+window_size+1, len(words))):
            if i != j and words[i] in tfidf_feature_names and words[j] in tfidf_feature_names:
                G.add_edge(i, j, weight=1.0)
    
    return G

def convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names):
    if len(G.nodes) == 0:
        # Return a dummy graph for empty input
        return Data(
            x=torch.zeros((1, len(tfidf_feature_names)), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long)
        )
    
    # Node features from TF-IDF
    x = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        if 'feature_idx' in node_data:
            feature_idx = node_data['feature_idx']
            x.append(tfidf_matrix[:, feature_idx].toarray().flatten())
        else:
            x.append(np.zeros(tfidf_matrix.shape[0]))
    
    if len(x) == 0:
        return Data(
            x=torch.zeros((1, len(tfidf_feature_names)), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long)
        )
    
    x = np.array(x)
    if x.shape[0] > 0:
        x = torch.FloatTensor(x)
    else:
        x = torch.zeros((1, tfidf_matrix.shape[0]), dtype=torch.float)
    
    # Edge indices
    edge_index = []
    for edge in G.edges():
        edge_index.append([edge[0], edge[1]])
        edge_index.append([edge[1], edge[0]])  # Add bidirectional edges
    
    if len(edge_index) > 0:
        edge_index = torch.LongTensor(edge_index).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

def main():
    print("Starting GNN optimization...")
    start_time = time.time()
    
    # Load the combined dataset
    combined_path = 'Dataset/combined_fake_news.csv'
    if not os.path.exists(combined_path):
        print(f"Combined dataset not found at {combined_path}. Please run combine_datasets.py first.")
        return
    
    df = pd.read_csv(combined_path)
    print(f"Loaded combined dataset with {df.shape[0]} samples")
    
    # Check for text column in the dataset
    text_columns = ['text', 'content', 'title', 'article_text']
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"No text column found in the dataset. Available columns: {df.columns}")
        return
    
    # Clean text
    print("Preprocessing text...")
    df['cleaned_text'] = df[text_col].apply(preprocess_text)
    
    # Keep only rows with non-empty text
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Create graph datasets
    print("Converting text to graphs...")
    train_data_list = []
    for i, (text, label) in tqdm(enumerate(zip(X_train, y_train)), total=len(X_train)):
        G = create_graph_from_text(text, tfidf_matrix, tfidf_feature_names)
        graph_data = convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names)
        graph_data.y = torch.tensor([label], dtype=torch.long)
        train_data_list.append(graph_data)
    
    test_data_list = []
    for i, (text, label) in tqdm(enumerate(zip(X_test, y_test)), total=len(X_test)):
        G = create_graph_from_text(text, tfidf_matrix, tfidf_feature_names)
        graph_data = convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names)
        graph_data.y = torch.tensor([label], dtype=torch.long)
        test_data_list.append(graph_data)
    
    # Create data loaders
    train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(input_dim=tfidf_matrix.shape[0], hidden_channels=64, output_dim=2).to(device)
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    print(f"Training GNN model on {device}...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print("\nGNN Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the results
    results = {
        'model': 'GNN',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'processing_time': time.time() - start_time
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('gnn_results.csv', index=False)
    print(f"Results saved to gnn_results.csv")
    
    print(f"GNN optimization completed in {time.time() - start_time:.2f} seconds")
    
    # Return results for later comparison
    return results

if __name__ == "__main__":
    main() 