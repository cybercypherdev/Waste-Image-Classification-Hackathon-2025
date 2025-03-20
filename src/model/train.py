#new
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data_chunks(data_dir):
    """Load data chunks from directory."""
    chunk_files = [f for f in os.listdir(data_dir) if f.startswith('chunk_') and f.endswith('.npz')]
    chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return [os.path.join(data_dir, f) for f in chunk_files]

def get_data_shape(chunk_files):
    """Get total number of samples and shape from chunks."""
    total_samples = 0
    sample_shape = None
    for chunk_file in chunk_files:
        with np.load(chunk_file) as data:
            total_samples += data['images'].shape[0]
            if sample_shape is None:
                sample_shape = data['images'].shape[1:]
    return total_samples, sample_shape

def batch_transform(scaler, X, batch_size=50):
    """Transform data in batches."""
    n_samples = X.shape[0]
    n_features = X.shape[1]
    X_transformed = np.empty((n_samples, n_features), dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_transformed[i:end] = scaler.transform(X[i:end])
    
    return X_transformed

def fit_scaler_on_chunks(scaler, chunk_files, batch_size=50):
    """Fit scaler on data chunks."""
    # First pass: compute mean
    mean_sum = 0
    n_samples = 0
    
    print("Computing mean...")
    for chunk_file in chunk_files:
        with np.load(chunk_file) as data:
            images = data['images'].astype(np.float32)
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size].reshape(min(batch_size, images.shape[0] - i), -1)
                mean_sum += np.sum(batch, axis=0)
                n_samples += batch.shape[0]
    
    mean = mean_sum / n_samples
    
    # Second pass: compute variance
    var_sum = 0
    print("Computing variance...")
    for chunk_file in chunk_files:
        with np.load(chunk_file) as data:
            images = data['images'].astype(np.float32)
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size].reshape(min(batch_size, images.shape[0] - i), -1)
                var_sum += np.sum((batch - mean) ** 2, axis=0)
    
    var = var_sum / n_samples
    scale = np.sqrt(var)
    
    # Set scaler parameters
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = var
    scaler.n_samples_seen_ = n_samples
    
    return scaler

def train_model_on_chunks(model, chunk_files, scaler, batch_size=50, max_samples_per_class=1000):
    """Train model on data chunks with sampling for memory efficiency."""
    if isinstance(model, LinearSVC):
        # For LinearSVC, collect balanced sample
        print("Collecting balanced sample for LinearSVC training...")
        sampled_X = []
        sampled_y = []
        class_counts = {}
        
        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {chunk_idx + 1}/{len(chunk_files)} for sampling...")
            with np.load(chunk_file) as data:
                X = data['images'].astype(np.float32)
                y = data['labels']
                
                for i in range(len(y)):
                    label = y[i]
                    if label not in class_counts:
                        class_counts[label] = 0
                    if class_counts[label] < max_samples_per_class:
                        X_sample = X[i].reshape(1, -1)
                        X_sample_scaled = scaler.transform(X_sample)
                        sampled_X.append(X_sample_scaled)
                        sampled_y.append(label)
                        class_counts[label] += 1
                
                # Check if we have enough samples
                if all(count >= max_samples_per_class for count in class_counts.values()):
                    break
        
        if sampled_X:
            print("Training LinearSVC on sampled data...")
            X_train = np.vstack(sampled_X)
            y_train = np.array(sampled_y)
            model.fit(X_train, y_train)
            del X_train
            del y_train
    else:
        # For SGDClassifier, use partial_fit
        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"Training on chunk {chunk_idx + 1}/{len(chunk_files)}...")
            with np.load(chunk_file) as data:
                X = data['images'].astype(np.float32)
                y = data['labels']
                
                for i in range(0, X.shape[0], batch_size):
                    batch_end = min(i + batch_size, X.shape[0])
                    X_batch = X[i:batch_end].reshape(batch_end - i, -1)
                    X_batch_scaled = scaler.transform(X_batch)
                    y_batch = y[i:batch_end]
                    
                    if i == 0 and chunk_idx == 0:
                        model.partial_fit(X_batch_scaled, y_batch, classes=np.unique(y))
                    else:
                        model.partial_fit(X_batch_scaled, y_batch)
    
    return model

def evaluate_model_on_chunks(model, chunk_files, scaler, batch_size=50):
    """Evaluate model on data chunks."""
    all_predictions = []
    all_true_labels = []
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        print(f"Evaluating on chunk {chunk_idx + 1}/{len(chunk_files)}...")
        with np.load(chunk_file) as data:
            X = data['images'].astype(np.float32)
            y = data['labels']
            
            for i in range(0, X.shape[0], batch_size):
                batch_end = min(i + batch_size, X.shape[0])
                X_batch = X[i:batch_end].reshape(batch_end - i, -1)
                X_batch_scaled = scaler.transform(X_batch)
                y_batch = y[i:batch_end]
                
                predictions = model.predict(X_batch_scaled)
                all_predictions.extend(predictions)
                all_true_labels.extend(y_batch)
    
    return np.array(all_predictions), np.array(all_true_labels)

def train_models():
    # Load data chunks
    print("Loading data chunks...")
    train_chunks = [f for f in os.listdir('data/train') if f.endswith('.npz')]
    train_chunks = [os.path.join('data/train', f) for f in train_chunks]
    test_chunks = [f for f in os.listdir('data/test') if f.endswith('.npz')]
    test_chunks = [os.path.join('data/test', f) for f in test_chunks]
    
    # Create model directory if it doesn't exist
    os.makedirs('src/model', exist_ok=True)
    
    # Initialize scaler
    print("Loading scaler...")
    scaler = StandardScaler()
    scaler = fit_scaler_on_chunks(scaler, train_chunks)
    
    # Train and evaluate SGD Classifier
    print("\nTraining SGD Classifier model...")
    sgd_model = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        random_state=42,
        learning_rate='optimal',
        tol=1e-3
    )
    sgd_model = train_model_on_chunks(sgd_model, train_chunks, scaler)
    
    print("\nEvaluating SGD Classifier...")
    sgd_pred, y_test = evaluate_model_on_chunks(sgd_model, test_chunks, scaler)
    sgd_acc = accuracy_score(y_test, sgd_pred)
    print(f"\nSGD Classifier Accuracy: {sgd_acc:.4f}")
    print("\nSGD Classifier Classification Report:")
    print(classification_report(y_test, sgd_pred))
    
    # Save SGD Classifier confusion matrix
    plt.figure(figsize=(8, 6))
    cm_sgd = confusion_matrix(y_test, sgd_pred)
    sns.heatmap(cm_sgd, annot=True, fmt='d', cmap='Blues')
    plt.title('SGD Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('src/model/sgd_confusion_matrix.png')
    plt.close()
    
    # Save SGD model and scaler first
    print("\nSaving SGD model and scaler...")
    joblib.dump(sgd_model, 'src/model/sgd_model.joblib')
    joblib.dump(scaler, 'src/model/scaler.joblib')
    
    # Free up memory
    del sgd_pred
    del sgd_model
    
    # Train and evaluate LinearSVC
    print("\nTraining LinearSVC model...")
    svm_model = LinearSVC(
        random_state=42,
        max_iter=5000,  # Increased from 2000 to 5000
        dual='auto',
        tol=1e-4  # Added smaller tolerance for better convergence
    )
    svm_model = train_model_on_chunks(svm_model, train_chunks, scaler)
    
    print("\nEvaluating LinearSVC...")
    svm_pred, y_test = evaluate_model_on_chunks(svm_model, test_chunks, scaler)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"\nLinearSVC Accuracy: {svm_acc:.4f}")
    print("\nLinearSVC Classification Report:")
    print(classification_report(y_test, svm_pred))
    
    # Save LinearSVC confusion matrix
    plt.figure(figsize=(8, 6))
    cm_svm = confusion_matrix(y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
    plt.title('LinearSVC Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('src/model/svm_confusion_matrix.png')
    plt.close()
    
    # Save LinearSVC model
    print("\nSaving LinearSVC model...")
    joblib.dump(svm_model, 'src/model/svm_model.joblib')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_models() 