import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "./"
TEST_DIR = os.path.join(DATA_DIR, "test/test/")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
MODEL_SAVE_DIR = "saved_models"
IMG_SIZE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

print(f"Using device: {DEVICE}")

# Load original training data to get label encoder
df_train = pd.read_csv(CSV_PATH)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['TARGET'])
num_classes = len(le.classes_)

print(f"Number of classes: {num_classes}")
print("Classes:", le.classes_)

# Test dataset
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()  # Ensure consistent ordering
        print(f"Found {len(self.image_files)} test images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        
        try:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback image
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            
        return image, img_name

# Test transforms
def get_test_transforms(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Model loading functions
def get_model(model_name, num_classes=20):
    if model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_model_checkpoint(model_path, model_name, device):
    """Load a model checkpoint"""
    model = get_model(model_name, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('best_f1', 0)

def predict_with_single_model(model, test_loader, device):
    """Make predictions with a single model"""
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            batch_preds = F.softmax(outputs, dim=1).cpu().numpy()
            predictions.append(batch_preds)
            image_names.extend(names)
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions, image_names

def predict_resnext_only():
    """Run predictions for ResNeXt models only"""
    print("🔮 Starting ResNeXt predictions...")
    
    # Create test dataset
    test_dataset = TestDataset(TEST_DIR, get_test_transforms(IMG_SIZE))
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    model_name = 'resnext'
    model_folder = os.path.join(MODEL_SAVE_DIR, model_name)
    
    if not os.path.exists(model_folder):
        print(f"❌ Model folder not found: {model_folder}")
        return
    
    # Get all model files for ResNeXt
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    model_files.sort()
    
    if not model_files:
        print(f"❌ No model files found in {model_folder}")
        return
    
    print(f"Found {len(model_files)} ResNeXt models: {model_files}")
    
    # Load models and get their F1 scores
    models_with_scores = []
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        try:
            model, f1_score = load_model_checkpoint(model_path, model_name, DEVICE)
            models_with_scores.append((model.to(DEVICE), f1_score, model_file))
            print(f"✅ Loaded {model_file}: F1 = {f1_score:.4f}")
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
            continue
    
    if not models_with_scores:
        print(f"❌ No models loaded successfully for {model_name}")
        return
    
    # Sort by F1 score (best first)
    models_with_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n📊 ResNeXt model ranking by F1 score:")
    for i, (_, f1, name) in enumerate(models_with_scores):
        print(f"  {i+1}. {name}: {f1:.4f}")
    
    # Make predictions with each model
    all_predictions = []
    image_names = None
    
    for i, (model, f1_score, model_file) in enumerate(models_with_scores):
        print(f"\n🔮 Predicting with {model_file}...")
        
        try:
            predictions, current_image_names = predict_with_single_model(model, test_loader, DEVICE)
            all_predictions.append(predictions)
            
            if image_names is None:
                image_names = current_image_names
            
            print(f"✅ Completed {model_file}: {predictions.shape}")
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error with {model_file}: {e}")
            continue
    
    if not all_predictions:
        print(f"❌ No predictions made for ResNeXt")
        return
    
    # Create averaged prediction for ResNeXt
    print(f"\n🔗 Creating averaged ResNeXt prediction...")
    
    # Average predictions across all ResNeXt models
    avg_predictions = np.mean(all_predictions, axis=0)
    predicted_classes = np.argmax(avg_predictions, axis=1)
    predicted_labels = le.inverse_transform(predicted_classes)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'ID': image_names,
        'TARGET': predicted_labels
    })
    submission_df = submission_df.sort_values('ID').reset_index(drop=True)
    
    # Save submission
    submission_path = "submission_resnext.csv"
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✅ Saved: {submission_path}")
    print(f"📊 ResNeXt Summary:")
    print(f"   Total images: {len(submission_df)}")
    print(f"   Models used: {len(all_predictions)}")
    print("   Predicted class distribution:")
    print(submission_df['TARGET'].value_counts().head(10))
    
    # Show first few predictions
    print(f"\n📋 First 10 predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n🎉 ResNeXt prediction completed!")
    print(f"📁 File saved: {submission_path}")

if __name__ == "__main__":
    print("🚀 Starting ResNeXt-only prediction...")
    
    # Check if test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory not found: {TEST_DIR}")
        print("Please ensure test images are in the correct directory")
        exit(1)
    
    # Check if ResNeXt model directory exists
    model_dir = os.path.join(MODEL_SAVE_DIR, 'resnext')
    if not os.path.exists(model_dir):
        print(f"❌ ResNeXt model directory not found: {model_dir}")
        exit(1)
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print(f"❌ No ResNeXt model files found in: {model_dir}")
        exit(1)
    
    print(f"✅ Found {len(model_files)} ResNeXt models")
    
    # Run ResNeXt predictions
    predict_resnext_only()


