import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import time
import os
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class ElectricEEL:
    """
    Electric EEL (Enhanced Evolutionary Learning) algorithm for fake news detection.
    This algorithm is inspired by the hunting behavior of electric eels.
    """
    
    def __init__(self, 
                 population_size=100, 
                 num_generations=50,
                 shock_intensity=0.8,
                 charge_decay=0.95,
                 mutation_rate=0.1,
                 feature_selection_ratio=0.7):
        self.population_size = population_size
        self.num_generations = num_generations
        self.shock_intensity = shock_intensity
        self.charge_decay = charge_decay
        self.mutation_rate = mutation_rate
        self.feature_selection_ratio = feature_selection_ratio
        self.best_solution = None
        self.best_fitness = 0
        self.weights = None
        
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
    
    def initialize_population(self, num_features):
        """Initialize a population of potential solutions (feature subsets)"""
        population = []
        for _ in range(self.population_size):
            # Create a binary mask for feature selection
            # 1 = feature is selected, 0 = feature is not selected
            solution = np.random.binomial(1, self.feature_selection_ratio, size=num_features)
            population.append(solution)
        return population
    
    def calculate_fitness(self, solution, X, y):
        """
        Calculate fitness score for a solution.
        In this case, fitness is the weighted sum of accuracy and feature reduction.
        """
        # Select features using the binary mask
        selected_features = solution > 0
        
        # If no features are selected, return 0 fitness
        if np.sum(selected_features) == 0:
            return 0
        
        # Extract selected features from X
        X_selected = X[:, selected_features]
        
        # Split data for quick evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42
        )
        
        # Train a simple model (weights) using the selected features
        # This is a simple linear model where weights are calculated
        # based on class-conditional averages
        class_0_avg = np.mean(X_train[y_train == 0], axis=0)
        class_1_avg = np.mean(X_train[y_train == 1], axis=0)
        
        # The weights are the difference between class averages
        weights = class_1_avg - class_0_avg
        
        # Make predictions using the weights
        y_pred = (X_test @ weights > 0).astype(int)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature reduction reward (penalty for using too many features)
        feature_reduction = 1 - (np.sum(selected_features) / len(solution))
        
        # Combine accuracy and feature reduction with a weight
        # Accuracy is more important (0.8) than feature reduction (0.2)
        fitness = 0.8 * accuracy + 0.2 * feature_reduction
        
        return fitness
    
    def electric_discharge(self, population, fitnesses):
        """
        Simulate electric eel's discharge to hunt prey.
        Better solutions have stronger discharge that affect nearby solutions.
        """
        new_population = []
        
        # Normalize fitnesses to use as discharge strengths
        max_fitness = max(fitnesses) if max(fitnesses) > 0 else 1
        discharge_strengths = [f / max_fitness for f in fitnesses]
        
        for i, solution in enumerate(population):
            # Each solution discharges with strength proportional to its fitness
            discharge_strength = discharge_strengths[i] * self.shock_intensity
            
            # Find nearby solutions (prey)
            for j, prey in enumerate(population):
                if i != j:
                    # Calculate distance between solution and prey
                    distance = np.sum(np.abs(solution - prey)) / len(solution)
                    
                    # The shock effect diminishes with distance
                    shock_effect = discharge_strength * np.exp(-distance)
                    
                    # Update prey based on shock effect
                    # If shock is strong, prey moves towards the solution
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
    
    def mutate(self, population):
        """Apply random mutations to the population"""
        for i in range(len(population)):
            # Each feature has a small chance to be flipped
            for j in range(len(population[i])):
                if random.random() < self.mutation_rate:
                    population[i][j] = 1 - population[i][j]  # Flip 0->1 or 1->0
        return population
    
    def charge_cycle(self, population, fitnesses):
        """Simulate the eel's charging cycle (selection)"""
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
        
        # Keep best solutions intact
        elite_size = int(0.1 * len(population))
        elite_indices = sorted_indices[:elite_size]
        elite_solutions = [population[i].copy() for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Add elite solutions
        new_population.extend(elite_solutions)
        
        # Fill the rest with weighted selection (higher fitness = higher chance)
        weights = np.array(fitnesses) / sum(fitnesses) if sum(fitnesses) > 0 else None
        
        while len(new_population) < len(population):
            if weights is None:
                idx = random.randint(0, len(population) - 1)
            else:
                idx = np.random.choice(len(population), p=weights)
            new_population.append(population[idx].copy())
        
        return new_population
    
    def fit(self, X, y):
        """Train the Electric EEL algorithm"""
        num_features = X.shape[1]
        
        # Initialize population
        population = self.initialize_population(num_features)
        
        # Track progress
        best_fitnesses = []
        
        # Main evolutionary loop
        for generation in tqdm(range(self.num_generations), desc="EEL Generations"):
            # Calculate fitness for each solution
            fitnesses = [self.calculate_fitness(solution, X, y) for solution in population]
            
            # Track the best solution
            max_fitness_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[max_fitness_idx]
            current_best_solution = population[max_fitness_idx].copy()
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution
                
            best_fitnesses.append(self.best_fitness)
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {self.best_fitness:.4f}, "
                      f"Features: {np.sum(self.best_solution)}/{num_features}")
            
            # Apply electric discharge (exploration)
            population = self.electric_discharge(population, fitnesses)
            
            # Apply mutation for further exploration
            population = self.mutate(population)
            
            # Apply charge cycle (selection)
            population = self.charge_cycle(population, fitnesses)
            
            # Reduce shock intensity over time (simulating eel getting tired)
            self.shock_intensity *= self.charge_decay
        
        # Train final weights using the best feature subset
        selected_features = self.best_solution > 0
        X_selected = X[:, selected_features]
        
        class_0_avg = np.mean(X_selected[y == 0], axis=0)
        class_1_avg = np.mean(X_selected[y == 1], axis=0)
        self.weights = class_1_avg - class_0_avg
        
        print(f"Electric EEL optimization complete.")
        print(f"Best solution has {np.sum(self.best_solution)} features out of {num_features}")
        print(f"Best fitness: {self.best_fitness:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using the best solution"""
        if self.best_solution is None or self.weights is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
        
        # Select features using the best solution
        selected_features = self.best_solution > 0
        X_selected = X[:, selected_features]
        
        # Make predictions using the trained weights
        y_pred = (X_selected @ self.weights > 0).astype(int)
        
        return y_pred

def main():
    print("Starting Electric EEL optimization...")
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
    
    # Create an instance of ElectricEEL
    eel = ElectricEEL(
        population_size=100,
        num_generations=50,
        shock_intensity=0.8,
        charge_decay=0.95,
        mutation_rate=0.1,
        feature_selection_ratio=0.7
    )
    
    # Clean text
    print("Preprocessing text...")
    df['cleaned_text'] = df[text_col].apply(eel.preprocess_text)
    
    # Keep only rows with non-empty text
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train the Electric EEL model
    print("Training Electric EEL model...")
    eel.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = eel.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("\nElectric EEL Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the results
    results = {
        'model': 'Electric_EEL',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'processing_time': time.time() - start_time,
        'selected_features': int(np.sum(eel.best_solution)),
        'total_features': X.shape[1]
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('electric_eel_results.csv', index=False)
    print(f"Results saved to electric_eel_results.csv")
    
    print(f"Electric EEL optimization completed in {time.time() - start_time:.2f} seconds")
    
    # Return results for later comparison
    return results

if __name__ == "__main__":
    main() 