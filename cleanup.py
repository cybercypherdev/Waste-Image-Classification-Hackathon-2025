import os
import shutil

def cleanup():
    # Directories to clean
    temp_dirs = [
        'data/temp',
        'data/train',
        'data/test'
    ]
    
    # Files to keep
    keep_files = [
        'src/model/sgd_model.joblib',
        'src/model/svm_model.joblib',
        'src/model/scaler.joblib',
        'src/model/sgd_confusion_matrix.png',
        'src/model/svm_confusion_matrix.png',
        'data/metadata.npy'
    ]
    
    print("Starting cleanup...")
    
    # Clean up temporary directories
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            print(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
    
    print("\nCleanup completed!")
    print("\nImportant files preserved:")
    for file in keep_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")

if __name__ == "__main__":
    cleanup() 