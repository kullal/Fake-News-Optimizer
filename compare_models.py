import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tabulate import tabulate

def load_results():
    """Load results from all model runs"""
    results = []
    
    # Check for GNN results
    if os.path.exists('gnn_results.csv'):
        gnn_results = pd.read_csv('gnn_results.csv')
        results.append(gnn_results)
    else:
        print("GNN results not found. Run gnn_optimization.py first.")
    
    # Check for Electric EEL results
    if os.path.exists('electric_eel_results.csv'):
        eel_results = pd.read_csv('electric_eel_results.csv')
        results.append(eel_results)
    else:
        print("Electric EEL results not found. Run electric_eel_optimization.py first.")
    
    # Check for Hybrid results
    if os.path.exists('hybrid_gnn_eel_results.csv'):
        hybrid_results = pd.read_csv('hybrid_gnn_eel_results.csv')
        results.append(hybrid_results)
    else:
        print("Hybrid GNN-EEL results not found. Run hybrid_gnn_eel_optimization.py first.")
    
    if not results:
        print("No results found. Run the optimization scripts first.")
        return None
    
    # Combine all results
    combined_results = pd.concat(results, ignore_index=True)
    
    return combined_results

def create_comparison_table(results_df):
    """Create a formatted comparison table of model metrics"""
    if results_df is None or results_df.empty:
        return "No results available for comparison."
    
    # Format the table for display
    table_data = []
    for _, row in results_df.iterrows():
        model_name = row['model']
        accuracy = f"{row['accuracy']:.4f}"
        precision = f"{row['precision']:.4f}"
        recall = f"{row['recall']:.4f}"
        f1_score = f"{row['f1_score']:.4f}"
        
        # Add feature selection info if available
        feature_info = ""
        if 'selected_features' in row and 'total_features' in row:
            feature_info = f"{row['selected_features']}/{row['total_features']}"
        
        # Add processing time
        proc_time = f"{row['processing_time'] / 60:.2f} min" if 'processing_time' in row else "N/A"
        
        table_data.append([model_name, accuracy, precision, recall, f1_score, feature_info, proc_time])
    
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Selected Features", "Processing Time"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    return table

def plot_performance_comparison(results_df):
    """Create bar charts comparing model performance"""
    if results_df is None or results_df.empty:
        print("No results available for plotting.")
        return
    
    # Set style
    sns.set(style="whitegrid")
    
    # Prepare metrics for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Color palette
    colors = sns.color_palette("viridis", len(results_df))
    
    # Create a bar plot for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Check if the metric exists in results
        if metric in results_df.columns:
            bars = ax.bar(results_df['model'], results_df[metric], color=colors)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.set_ylim(0, 1.0)  # Metrics are between 0 and 1
            ax.set_ylabel('Score', fontsize=10)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("Performance comparison plot saved as 'model_comparison_metrics.png'")
    
    # Processing time comparison
    if 'processing_time' in results_df.columns:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['model'], results_df['processing_time'] / 60, color=sns.color_palette("viridis", len(results_df)))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f} min', ha='center', va='bottom', fontsize=9)
        
        plt.title('Processing Time Comparison', fontsize=14)
        plt.ylabel('Time (minutes)', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig('model_comparison_time.png', dpi=300, bbox_inches='tight')
        print("Processing time comparison plot saved as 'model_comparison_time.png'")

def create_radar_chart(results_df):
    """Create a radar chart for visual comparison of models"""
    if results_df is None or results_df.empty:
        print("No results available for radar chart.")
        return
    
    # Metrics to include in radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Check if all metrics exist
    for metric in metrics:
        if metric not in results_df.columns:
            print(f"Metric {metric} not found in results. Skipping radar chart.")
            return
    
    # Number of variables
    N = len(metrics)
    
    # Create angle for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Color palette
    colors = sns.color_palette("viridis", len(results_df))
    
    # Add each model
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['model']
        
        # Extract metric values
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add feature labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Adjust the starting position
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set y limits
    plt.ylim(0, 1)
    
    # Add title
    plt.title('Model Performance Comparison', size=16, y=1.1)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig('model_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("Radar chart saved as 'model_comparison_radar.png'")

def main():
    print("Comparing model performance...")
    
    # Load results
    results_df = load_results()
    
    if results_df is not None and not results_df.empty:
        # Create comparison table
        table = create_comparison_table(results_df)
        print("\nModel Performance Comparison:")
        print(table)
        
        # Save table to file
        with open('model_comparison_table.txt', 'w') as f:
            f.write(table)
        print("Comparison table saved to 'model_comparison_table.txt'")
        
        # Create plots
        plot_performance_comparison(results_df)
        create_radar_chart(results_df)
        
        # Determine the best model based on F1 score
        if 'f1_score' in results_df.columns:
            best_model_idx = results_df['f1_score'].idxmax()
            best_model = results_df.loc[best_model_idx, 'model']
            best_f1 = results_df.loc[best_model_idx, 'f1_score']
            print(f"\nBest performing model based on F1 score: {best_model} (F1 = {best_f1:.4f})")
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main() 