## ![favicon-16x16](https://github.com/user-attachments/assets/8237d9de-cac8-43dc-957e-947c799d3cba) Waste Image Classification System
by Ephraim Maina

A machine learning system that classifies waste images into different categories using SGD Classifier and Linear SVM models.
![Screenshot 2025-03-20 171640](https://github.com/user-attachments/assets/a1c0bce5-3d78-4259-9987-0ee63b0997c3)

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
git clone https://github.com/cybercypherdev/Waste-Image-Classification-Hackathon-2025.git
cd Waste-Image-Classification-Hackathon-2025
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
or
```bash
pip install -r requiremets.txt

```

4. Prepare your dataset:
   - Download data sets from this link 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n3gtgm9jxj-2.zip'
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
python run_api.py
```

The server will run at `http://localhost:8000`

## Using the System

### Web Interface
1. Open `src/ui/index.html` in a web browser if you are running it localy
2. Upload an image using the interface
3. View the classification results and confidence scores
   ### How to use the web Interface
![Screenshot 2025-03-20 171715](https://github.com/user-attachments/assets/360c1104-3f9a-4b32-ac46-a9426f117f2c)

   ### Video Guide for using the web interface
   

https://github.com/user-attachments/assets/67d69fc9-3c96-4caa-bde3-8ae7c2a9a28f



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
  ![Screenshot 2025-03-20 001210](https://github.com/user-attachments/assets/40bffce5-e353-4b8e-8358-d4ab614051b4)

- Classification reports
  ![Screenshot 2025-03-20 001326](https://github.com/user-attachments/assets/67b178cd-8db1-468b-af22-aa60263e243a)

- Confusion matrices (saved as PNG files)
  ![sgd_confusion_matrix](https://github.com/user-attachments/assets/615b9326-5503-43a1-af76-5dfbbb5aa0dc)
![svm_confusion_matrix](https://github.com/user-attachments/assets/18a9dd6e-c3b3-4b61-8aef-324edc4e5a6b)
]
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
The main issue has been memory management and file location errors
![sdsd](https://github.com/user-attachments/assets/041dd219-bf95-44b8-a083-83542518fed3)

1. Memory errors:
   - Reduce batch_size in data_loader.py
   - Decrease max_samples_per_class in train.py
     ![Screenshot 2025-03-20 001352](https://github.com/user-attachments/assets/5c5d0668-688f-42f9-8e05-0d9d0c8b40fe)


2. Model convergence:
   - Increase max_iter in LinearSVC
   - Adjust tol parameter
   - Modify learning_rate in SGDClassifier

3. File not found errors:
   - Ensure correct directory structure
   - Check file permissions
   - Verify model files are saved correctly

## License

MIT license
