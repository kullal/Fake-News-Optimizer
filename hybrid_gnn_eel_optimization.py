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
import random

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class HybridGNNEEL:
    """
    Hybrid model combining Graph Neural Network (GNN) with Electric EEL for fake news detection.
    The Electric EEL component is used for feature selection and optimization,
    while the GNN component is used for graph-based learning.
    """
    
    def __init__(self, 
                 gnn_hidden_channels=64, 
                 eel_population_size=100,
                 eel_num_generations=30,
                 eel_shock_intensity=0.8,
                 eel_charge_decay=0.95,
                 eel_mutation_rate=0.1,
                 eel_feature_selection_ratio=0.7):
        # GNN parameters
        self.gnn_hidden_channels = gnn_hidden_channels
        
        # EEL parameters
        self.eel_population_size = eel_population_size
        self.eel_num_generations = eel_num_generations
        self.eel_shock_intensity = eel_shock_intensity
        self.eel_charge_decay = eel_charge_decay
        self.eel_mutation_rate = eel_mutation_rate
        self.eel_feature_selection_ratio = eel_feature_selection_ratio
        
        # Optimization trackers
        self.best_solution = None
        self.best_fitness = 0
        self.gnn_model = None

    def preprocess_text(self, text):
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
    
    def create_graph_from_text(self, text, tfidf_matrix, tfidf_feature_names, selected_features=None):
        """Create a graph representation of the text for GNN processing"""
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
                # If we have a feature selection mask, use it
                if selected_features is not None and not selected_features[feature_idx]:
                    continue
                G.add_node(i, word=word, feature_idx=feature_idx)
        
        # Add edges between words that co-occur within a window
        window_size = 5
        for i in range(len(words)):
            for j in range(i+1, min(i+window_size+1, len(words))):
                if i != j and words[i] in tfidf_feature_names and words[j] in tfidf_feature_names:
                    # If using feature selection, check both words
                    if selected_features is not None:
                        idx_i = tfidf_feature_names.index(words[i])
                        idx_j = tfidf_feature_names.index(words[j])
                        if not selected_features[idx_i] or not selected_features[idx_j]:
                            continue
                    G.add_edge(i, j, weight=1.0)
        
        return G
    
    def convert_to_pytorch_geometric(self, G, tfidf_matrix, tfidf_feature_names, selected_features=None):
        """Convert networkx graph to PyTorch Geometric data format"""
        if len(G.nodes) == 0:
            # Return a dummy graph for empty input
            feature_dim = len(tfidf_feature_names)
            if selected_features is not None:
                feature_dim = int(np.sum(selected_features))
            return Data(
                x=torch.zeros((1, feature_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        
        # Node features from TF-IDF
        x = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            if 'feature_idx' in node_data:
                feature_idx = node_data['feature_idx']
                # If using feature selection, get only selected features
                if selected_features is not None:
                    features = tfidf_matrix[:, feature_idx].toarray().flatten()
                    x.append(features[selected_features])
                else:
                    x.append(tfidf_matrix[:, feature_idx].toarray().flatten())
            else:
                feature_dim = tfidf_matrix.shape[0]
                if selected_features is not None:
                    feature_dim = int(np.sum(selected_features))
                x.append(np.zeros(feature_dim))
        
        if len(x) == 0:
            feature_dim = len(tfidf_feature_names)
            if selected_features is not None:
                feature_dim = int(np.sum(selected_features))
            return Data(
                x=torch.zeros((1, feature_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        
        x = np.array(x)
        if x.shape[0] > 0:
            x = torch.FloatTensor(x)
        else:
            feature_dim = tfidf_matrix.shape[0]
            if selected_features is not None:
                feature_dim = int(np.sum(selected_features))
            x = torch.zeros((1, feature_dim), dtype=torch.float)
        
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
    
    def initialize_eel_population(self, num_features):
        """Initialize a population of potential feature selection solutions"""
        population = []
        for _ in range(self.eel_population_size):
            # Create a binary mask for feature selection (1 = select, 0 = ignore)
            solution = np.random.binomial(1, self.eel_feature_selection_ratio, size=num_features)
            population.append(solution)
        return population
    
    def eel_calculate_fitness(self, solution, X, y, tfidf_matrix, tfidf_feature_names, text_data):
        """
        Calculate fitness for a feature selection solution using a small GNN model
        """
        # Select features using the binary mask
        selected_features = solution > 0
        
        # If no features are selected, return 0 fitness
        if np.sum(selected_features) == 0:
            return 0
        
        # Create a small subset for quick evaluation
        indices = np.random.choice(len(text_data), min(100, len(text_data)), replace=False)
        X_sample = [text_data[i] for i in indices]
        y_sample = [y[i] for i in indices]
        
        # Create graph datasets using selected features
        data_list = []
        for text, label in zip(X_sample, y_sample):
            G = self.create_graph_from_text(text, tfidf_matrix, tfidf_feature_names, selected_features)
            graph_data = self.convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names, selected_features)
            graph_data.y = torch.tensor([label], dtype=torch.long)
            data_list.append(graph_data)
        
        # If no valid graphs, return 0 fitness
        if len(data_list) == 0:
            return 0
        
        # Create data loader
        loader = DataLoader(data_list, batch_size=min(32, len(data_list)), shuffle=True)
        
        # Create a small GNN model for evaluation
        input_dim = np.sum(selected_features)
        if input_dim == 0:
            return 0
            
        device = torch.device('cpu')  # Use CPU for quick evaluation
        
        # Define a simple GNN model for evaluation
        class SimpleGNN(torch.nn.Module):
            def __init__(self, input_dim, hidden_channels=32, output_dim=2):
                super(SimpleGNN, self).__init__()
                self.conv = GCNConv(input_dim, hidden_channels)
                self.fc = nn.Linear(hidden_channels, output_dim)
                
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = self.conv(x, edge_index)
                x = F.relu(x)
                x = global_mean_pool(x, batch)
                x = self.fc(x)
                return F.log_softmax(x, dim=1)
        
        model = SimpleGNN(input_dim=input_dim, hidden_channels=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Quick training
        model.train()
        for epoch in range(5):  # Just a few epochs for quick evaluation
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
        
        # Quick evaluation
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Feature reduction reward
        feature_reduction = 1 - (np.sum(selected_features) / len(solution))
        
        # Combine accuracy and feature reduction
        fitness = 0.8 * accuracy + 0.2 * feature_reduction
        
        return fitness
    
    def eel_electric_discharge(self, population, fitnesses):
        """
        Simulate electric eel's discharge to optimize feature selection
        """
        # Normalize fitnesses to use as discharge strengths
        max_fitness = max(fitnesses) if max(fitnesses) > 0 else 1
        discharge_strengths = [f / max_fitness for f in fitnesses]
        
        for i, solution in enumerate(population):
            # Each solution discharges with strength proportional to its fitness
            discharge_strength = discharge_strengths[i] * self.eel_shock_intensity
            
            # Find nearby solutions (prey)
            for j, prey in enumerate(population):
                if i != j:
                    # Calculate distance between solution and prey
                    distance = np.sum(np.abs(solution - prey)) / len(solution)
                    
                    # The shock effect diminishes with distance
                    shock_effect = discharge_strength * np.exp(-distance)
                    
                    # Update prey based on shock effect
                    if random.random() < shock_effect:
                        # Get random positions to modify
                        positions = np.random.choice(
                            len(prey),
                            size=int(shock_effect * len(prey)),
                            replace=False
                        )
                        
                        # Move prey towards the solution at selected positions
                        for pos in positions:
                            prey[pos] = solution[pos]
                        
                        # Store the updated prey
                        population[j] = prey
        
        return population
    
    def eel_mutate(self, population):
        """Apply random mutations to the feature selection population"""
        for i in range(len(population)):
            # Each feature has a small chance to be flipped
            for j in range(len(population[i])):
                if random.random() < self.eel_mutation_rate:
                    population[i][j] = 1 - population[i][j]  # Flip 0->1 or 1->0
        return population
    
    def eel_charge_cycle(self, population, fitnesses):
        """Simulate the eel's charging cycle (selection)"""
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
        
        # Keep best solutions intact
        elite_size = max(1, int(0.1 * len(population)))
        elite_indices = sorted_indices[:elite_size]
        elite_solutions = [population[i].copy() for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Add elite solutions
        new_population.extend(elite_solutions)
        
        # Fill the rest with weighted selection
        weights = np.array(fitnesses) / sum(fitnesses) if sum(fitnesses) > 0 else None
        
        while len(new_population) < len(population):
            if weights is None:
                idx = random.randint(0, len(population) - 1)
            else:
                idx = np.random.choice(len(population), p=weights)
            new_population.append(population[idx].copy())
        
        return new_population
    
    def optimize_features(self, X, y, tfidf_matrix, tfidf_feature_names, text_data):
        """
        Use the Electric EEL algorithm to find optimal feature subset
        """
        print("Starting Electric EEL feature optimization...")
        num_features = tfidf_matrix.shape[1]
        
        # Initialize population
        population = self.initialize_eel_population(num_features)
        
        # Track progress
        best_fitnesses = []
        
        # Main EEL optimization loop
        for generation in tqdm(range(self.eel_num_generations), desc="EEL Feature Selection"):
            # Calculate fitness for each solution
            fitnesses = [
                self.eel_calculate_fitness(solution, X, y, tfidf_matrix, tfidf_feature_names, text_data) 
                for solution in population
            ]
            
            # Track the best solution
            max_fitness_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[max_fitness_idx]
            current_best_solution = population[max_fitness_idx].copy()
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution
                
            best_fitnesses.append(self.best_fitness)
            
            # Print progress every 5 generations
            if generation % 5 == 0:
                selected_count = np.sum(self.best_solution) if self.best_solution is not None else 0
                print(f"Generation {generation}, Best Fitness: {self.best_fitness:.4f}, "
                      f"Selected Features: {selected_count}/{num_features}")
            
            # Apply electric discharge (exploration)
            population = self.eel_electric_discharge(population, fitnesses)
            
            # Apply mutation for further exploration
            population = self.eel_mutate(population)
            
            # Apply charge cycle (selection)
            population = self.eel_charge_cycle(population, fitnesses)
            
            # Reduce shock intensity over time
            self.eel_shock_intensity *= self.eel_charge_decay
        
        print(f"EEL feature optimization complete.")
        print(f"Selected {np.sum(self.best_solution)} features out of {num_features}")
        
        return self.best_solution
    
    def build_gnn_model(self, input_dim):
        """Create the GNN component of the hybrid model"""
        class GNNModel(torch.nn.Module):
            def __init__(self, input_dim, hidden_channels, output_dim=2):
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
        
        return GNNModel(input_dim=input_dim, hidden_channels=self.gnn_hidden_channels)
    
    def fit(self, X, y, text_data, tfidf_matrix, tfidf_feature_names):
        """
        Train the hybrid GNN-EEL model:
        1. Use Electric EEL to select optimal features
        2. Build graph datasets using selected features
        3. Train a GNN on the optimized graph data
        """
        # Step 1: EEL feature selection
        selected_features = self.optimize_features(X, y, tfidf_matrix, tfidf_feature_names, text_data)
        
        # Apply feature selection
        selected_mask = selected_features > 0
        X_selected = X[:, selected_mask]
        
        # Split the data
        X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            X_selected, y, text_data, test_size=0.2, random_state=42
        )
        
        # Step 2: Build graph datasets using selected features
        print("Converting text to optimized graphs...")
        train_data_list = []
        for i, (text, label) in tqdm(enumerate(zip(text_train, y_train)), total=len(text_train)):
            G = self.create_graph_from_text(text, tfidf_matrix, tfidf_feature_names, selected_mask)
            graph_data = self.convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names, selected_mask)
            graph_data.y = torch.tensor([label], dtype=torch.long)
            train_data_list.append(graph_data)
        
        test_data_list = []
        for i, (text, label) in tqdm(enumerate(zip(text_test, y_test)), total=len(text_test)):
            G = self.create_graph_from_text(text, tfidf_matrix, tfidf_feature_names, selected_mask)
            graph_data = self.convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names, selected_mask)
            graph_data.y = torch.tensor([label], dtype=torch.long)
            test_data_list.append(graph_data)
        
        # Create data loaders
        train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)
        
        # Step 3: Build and train the GNN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = np.sum(selected_mask)
        self.gnn_model = self.build_gnn_model(input_dim).to(device)
        
        # Define the optimizer
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        # Training loop
        print(f"Training GNN component on {device}...")
        self.gnn_model.train()
        for epoch in range(10):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = self.gnn_model(batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        
        # Evaluation on test set
        print("Evaluating hybrid model...")
        self.gnn_model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(device)
                out = self.gnn_model(batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        print("\nHybrid GNN-EEL Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Return metrics for comparison
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'selected_features': int(np.sum(selected_mask)),
            'total_features': len(selected_mask)
        }
    
    def predict(self, text_data, tfidf_matrix, tfidf_feature_names):
        """Make predictions using the trained hybrid model"""
        if self.gnn_model is None or self.best_solution is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_model.eval()
        selected_mask = self.best_solution > 0
        
        # Convert text data to graphs
        data_list = []
        for text in text_data:
            G = self.create_graph_from_text(text, tfidf_matrix, tfidf_feature_names, selected_mask)
            graph_data = self.convert_to_pytorch_geometric(G, tfidf_matrix, tfidf_feature_names, selected_mask)
            data_list.append(graph_data)
        
        # Create a data loader
        loader = DataLoader(data_list, batch_size=64, shuffle=False)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.gnn_model(batch)
                pred = out.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        
        return predictions

def main():
    print("Starting Hybrid GNN-EEL optimization...")
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
    
    # Create the hybrid model
    hybrid_model = HybridGNNEEL(
        gnn_hidden_channels=64,
        eel_population_size=50,  # Reduced for faster execution
        eel_num_generations=20,  # Reduced for faster execution
        eel_shock_intensity=0.8,
        eel_charge_decay=0.95,
        eel_mutation_rate=0.1,
        eel_feature_selection_ratio=0.7
    )
    
    # Clean text
    print("Preprocessing text...")
    df['cleaned_text'] = df[text_col].apply(hybrid_model.preprocess_text)
    
    # Keep only rows with non-empty text
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label'].values
    text_data = df['cleaned_text'].values
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Train the hybrid model
    print("Training hybrid GNN-EEL model...")
    results = hybrid_model.fit(X, y, text_data, tfidf_vectorizer.transform(text_data), tfidf_feature_names)
    
    # Add model name and timing info to results
    results['model'] = 'Hybrid_GNN_EEL'
    results['processing_time'] = time.time() - start_time
    
    # Save the results
    results_df = pd.DataFrame([results])
    results_df.to_csv('hybrid_gnn_eel_results.csv', index=False)
    print(f"Results saved to hybrid_gnn_eel_results.csv")
    
    print(f"Hybrid GNN-EEL optimization completed in {time.time() - start_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    main() 