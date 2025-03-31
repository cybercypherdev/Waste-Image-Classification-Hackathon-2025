from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import joblib
import os
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kind-glacier-09583601e.6.azurestaticapps.net"],  # Specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Ensure CORS headers are included in responses
)
# Mount static files
app.mount("/static", StaticFiles(directory=r"C:\Users\Set a Bar\Desktop\Hackerthon\src\ui"), name="static")

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "svm_model.joblib")
scaler_path = os.path.join(os.path.dirname(__file__), "..", "model", "scaler.joblib")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("="*50)
    print("Model and scaler loaded successfully")
    print(f"Model classes: {model.classes_}")
    print(f"Model type: {type(model)}")
    print("="*50)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

def preprocess_image(image):
    try:
        # Convert BGR to RGB to match training preprocessing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256 while keeping RGB channels
        resized = cv2.resize(image, (256, 256))
        
        # Flatten and normalize
        flattened = resized.flatten() / 255.0
        
        # Ensure the shape is correct (196608 features = 256x256x3)
        if flattened.shape[0] != 196608:
            raise ValueError(f"Unexpected feature shape: {flattened.shape}")
            
        return flattened
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        raise

@app.get("/")
async def read_root():
    return {"message": "Welcome to Waste Classification API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("\n" + "="*50)
        print(f"Processing file: {file.filename}")
        
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Error: Invalid image file")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        
        print(f"Input image shape: {image.shape}")
        
        # Preprocess the image
        features = preprocess_image(image)
        print(f"Preprocessed features shape: {features.shape}")
        
        # Scale features
        features_scaled = scaler.transform([features])
        print(f"Scaled features shape: {features_scaled.shape}")
        
        # Get direct class prediction
        predicted_class = model.predict([features_scaled[0]])[0]
        print(f"Direct model prediction: {predicted_class}")
        
        # For LinearSVC, we'll use decision_function to get confidence scores
        decision_scores = model.decision_function([features_scaled[0]])
        print(f"Decision scores: {decision_scores}")
        
        # Convert decision scores to pseudo-probabilities using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        confidence_score = sigmoid(abs(decision_scores[0]))
        print(f"Confidence score: {confidence_score}")
        
        # Get class labels
        class_labels = model.classes_
        print(f"Model class labels: {class_labels}")
        
        # Create predictions dictionary
        if decision_scores[0] > 0:  # Positive score means second class (recyclable)
            predictions = {
                "Organic": float(1 - confidence_score),
                "Recyclable": float(confidence_score)
            }
        else:  # Negative score means first class (organic)
            predictions = {
                "Organic": float(confidence_score),
                "Recyclable": float(1 - confidence_score)
            }
        
        # Convert class name to standardized format
        final_prediction = "Organic" if predicted_class == "organic" else "Recyclable"
        
        print(f"Final prediction: {final_prediction}")
        print(f"Confidence: {float(confidence_score):.4f}")
        print(f"Full predictions: {predictions}")
        print("="*50 + "\n")
        
        return {
            "predictions": predictions,
            "prediction": final_prediction,
            "probability": float(confidence_score),
            "debug": {
                "class_labels": class_labels.tolist(),
                "decision_score": float(decision_scores[0]),
                "filename": file.filename
            }
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))  # Azure sets the port dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
