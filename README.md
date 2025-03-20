# Waste Image Classification System

A machine learning system that classifies waste images into different categories using SGD Classifier and Linear SVM models.

## Project Structure

```
├── data/
│   ├── train/          # Training data chunks
│   ├── test/           # Testing data chunks
│   └── waste_dataset/  # Raw image dataset
├── src/
│   ├── api/
│   │   └── app.py      # FastAPI server for predictions
│   ├── model/
│   │   ├── train.py    # Model training script
│   │   ├── sgd_model.joblib    # Trained SGD Classifier
│   │   ├── svm_model.joblib    # Trained Linear SVM
│   │   └── scaler.joblib       # Fitted StandardScaler
│   ├── preprocessing/
│   │   └── data_loader.py      # Data preprocessing script
│   └── ui/
│       └── index.html  # Web interface
|        |_app.js
|        |_style.css
└── run_pipeline.py     # Main pipeline script
|_run_api.py  #starts the API & Backend to serve User interface
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd waste-classification
```

2. Create a Python virtual environment (Python 3.8+ recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install scikit-learn numpy pandas matplotlib seaborn joblib pillow fastapi uvicorn python-multipart
```

4. Prepare your dataset:
   - Place your waste images in `data/waste_dataset/`
   - Images should be organized in category folders (e.g., organic, recyclable, etc.)

## Training the Models

1. Run the complete pipeline:
```bash
python run_pipeline.py
```

This will:
- Preprocess the images
- Split data into training and testing sets
- Train the SGD Classifier and Linear SVM models
- Generate evaluation metrics and confusion matrices
- Save the trained models and scaler

The pipeline uses memory-efficient techniques:
- Processes images in small batches
- Uses float32 data type
- Implements chunked data processing
- Employs balanced sampling for SVM training

## Starting the API Server

1. Start the FastAPI server:
```bash
python src/api/app.py
```

The server will run at `http://localhost:8000`

## Using the System

### Web Interface
1. Open `src/ui/index.html` in a web browser
2. Upload an image using the interface
3. View the classification results and confidence scores

### API Endpoints
- `POST /predict`: Submit an image for classification
  ```bash
  curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"
  ```
- `GET /health`: Check API status
  ```bash
  curl "http://localhost:8000/health"
  ```

## Model Details

### SGD Classifier
- Uses logistic regression loss function
- Trained incrementally using partial_fit
- Optimized for memory efficiency
- Suitable for large-scale learning

### Linear SVM
- Uses LinearSVC implementation
- Trained on balanced sample of data
- Memory-efficient alternative to kernel SVM
- Good performance on high-dimensional data

## Performance Metrics

The system generates:
- Accuracy scores for both models
- Classification reports
- Confusion matrices (saved as PNG files)
- Model performance comparisons

## Memory Optimization

The system implements several memory optimization techniques:
1. Batch processing of images (25 images per batch)
2. Chunked data storage and loading
3. Incremental model training
4. Balanced sampling for SVM (1000 samples per class)
5. Explicit memory cleanup
6. Float32 data type usage

## Troubleshooting

Common issues and solutions:
1. Memory errors:
   - Reduce batch_size in data_loader.py
   - Decrease max_samples_per_class in train.py

2. Model convergence:
   - Increase max_iter in LinearSVC
   - Adjust tol parameter
   - Modify learning_rate in SGDClassifier

3. File not found errors:
   - Ensure correct directory structure
   - Check file permissions
   - Verify model files are saved correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT license