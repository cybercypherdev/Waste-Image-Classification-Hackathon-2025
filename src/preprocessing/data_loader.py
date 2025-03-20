import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import joblib
import h5py

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess a single image."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values and convert to float32
        img = (img / 255.0).astype(np.float32)
        
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_and_save_batch(batch_paths, batch_labels, batch_idx, output_dir):
    """Process a batch of images and save them to a temporary file."""
    batch_images = []
    valid_labels = []
    
    for img_path, label in zip(batch_paths, batch_labels):
        img = preprocess_image(img_path)
        if img is not None:
            batch_images.append(img)
            valid_labels.append(label)
    
    if batch_images:
        # Save batch to temporary file
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_labels = np.array(valid_labels)
        
        # Save to temporary file
        temp_file = os.path.join(output_dir, f'batch_{batch_idx}.npz')
        np.savez_compressed(temp_file, images=batch_images, labels=batch_labels)
        return len(batch_images), temp_file
    return 0, None

def load_dataset(data_dir, batch_size=25):
    """Load and preprocess the entire dataset in batches."""
    # Create output directory
    output_dir = 'data/temp'
    os.makedirs(output_dir, exist_ok=True)
    
    total_images = 0
    batch_idx = 0
    temp_files = []
    
    print("Loading dataset...")
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            print(f"Processing {label}...")
            image_paths = [os.path.join(label_dir, img_name) for img_name in os.listdir(label_dir)]
            labels = [label] * len(image_paths)
            
            # Process images in smaller batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                processed_count, temp_file = process_and_save_batch(batch_paths, batch_labels, batch_idx, output_dir)
                if processed_count > 0:
                    total_images += processed_count
                    temp_files.append(temp_file)
                    batch_idx += 1
                
                print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images in {label}")
                
                # Save progress every 500 images
                if total_images % 500 == 0:
                    print(f"Total images processed so far: {total_images}")
    
    return output_dir, temp_files, total_images

def save_split_data(temp_dir, temp_files, total_images, train_ratio=0.8, chunk_size=250):
    """Save train and test splits from temporary files."""
    # Calculate split sizes
    n_train = int(total_images * train_ratio)
    n_test = total_images - n_train
    
    # Create output directories
    train_dir = 'data/train'
    test_dir = 'data/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Shuffle files
    np.random.shuffle(temp_files)
    
    # Split files
    train_files = temp_files[:int(len(temp_files) * train_ratio)]
    test_files = temp_files[int(len(temp_files) * train_ratio):]
    
    # Process training data in chunks
    print("Saving training data...")
    chunk_idx = 0
    train_images = []
    train_labels = []
    train_total = 0
    
    for file in train_files:
        try:
            data = np.load(file)
            train_images.append(data['images'])
            train_labels.append(data['labels'])
            data.close()
            
            # Save chunk when size exceeds chunk_size
            total_in_memory = sum(x.shape[0] for x in train_images)
            if total_in_memory >= chunk_size:
                images = np.concatenate(train_images, axis=0)
                labels = np.concatenate(train_labels, axis=0)
                np.savez_compressed(
                    os.path.join(train_dir, f'chunk_{chunk_idx}.npz'),
                    images=images, labels=labels
                )
                train_total += images.shape[0]
                chunk_idx += 1
                train_images = []
                train_labels = []
                print(f"Saved training chunk {chunk_idx}, total images: {train_total}")
            
            # Clean up temp file
            try:
                os.remove(file)
            except OSError:
                print(f"Warning: Could not remove {file}")
        except Exception as e:
            print(f"Warning: Error processing {file}: {str(e)}")
    
    # Save remaining training data
    if train_images:
        images = np.concatenate(train_images, axis=0)
        labels = np.concatenate(train_labels, axis=0)
        np.savez_compressed(
            os.path.join(train_dir, f'chunk_{chunk_idx}.npz'),
            images=images, labels=labels
        )
        train_total += images.shape[0]
        chunk_idx += 1
    
    # Process test data in chunks
    print("Saving test data...")
    chunk_idx = 0
    test_images = []
    test_labels = []
    test_total = 0
    
    for file in test_files:
        try:
            data = np.load(file)
            test_images.append(data['images'])
            test_labels.append(data['labels'])
            data.close()
            
            # Save chunk when size exceeds chunk_size
            total_in_memory = sum(x.shape[0] for x in test_images)
            if total_in_memory >= chunk_size:
                images = np.concatenate(test_images, axis=0)
                labels = np.concatenate(test_labels, axis=0)
                np.savez_compressed(
                    os.path.join(test_dir, f'chunk_{chunk_idx}.npz'),
                    images=images, labels=labels
                )
                test_total += images.shape[0]
                chunk_idx += 1
                test_images = []
                test_labels = []
                print(f"Saved test chunk {chunk_idx}, total images: {test_total}")
            
            # Clean up temp file
            try:
                os.remove(file)
            except OSError:
                print(f"Warning: Could not remove {file}")
        except Exception as e:
            print(f"Warning: Error processing {file}: {str(e)}")
    
    # Save remaining test data
    if test_images:
        images = np.concatenate(test_images, axis=0)
        labels = np.concatenate(test_labels, axis=0)
        np.savez_compressed(
            os.path.join(test_dir, f'chunk_{chunk_idx}.npz'),
            images=images, labels=labels
        )
        test_total += images.shape[0]
        chunk_idx += 1
    
    try:
        # Clean up temporary directory if empty
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except OSError:
        print(f"Warning: Could not remove temporary directory {temp_dir}")
    
    # Save metadata
    metadata = {
        'n_train': train_total,
        'n_test': test_total,
        'total_images': train_total + test_total,
        'n_train_chunks': chunk_idx,
        'n_test_chunks': chunk_idx
    }
    np.save('data/metadata.npy', metadata)
    
    return metadata

def main():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('src/model', exist_ok=True)
    
    # Use existing dataset
    data_dir = "data/waste_dataset"
    
    # Load and preprocess dataset
    temp_dir, temp_files, total_images = load_dataset(data_dir)
    
    # Save split data
    metadata = save_split_data(temp_dir, temp_files, total_images)
    
    print("Dataset preprocessing completed!")
    print(f"Total images processed: {metadata['total_images']}")
    print(f"Training set size: {metadata['n_train']}")
    print(f"Testing set size: {metadata['n_test']}")

if __name__ == "__main__":
    main() 