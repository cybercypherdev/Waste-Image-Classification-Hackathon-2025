import os
import sys
from src.preprocessing.data_loader import main as preprocess_data
from src.model.train import train_models

def main():
    print("Starting data preprocessing...")
    preprocess_data()
    
    print("\nStarting model training...")
    train_models()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 