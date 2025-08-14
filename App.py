rom flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import kaggle
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
from recommend import get_recommendations  # Import from recommendations module

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Kaggle Dataset Configuration
KAGGLE_DATASET = 'irkaal/indian-food-nutrition-data'
DATA_DIR = Path('data')
DATASET_PATH = DATA_DIR / 'indian_food.csv'

# File Upload Config
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_food_dataset():
    """Load food dataset with expanded local options"""
    expanded_dataset = [
        # Rice Dishes
        {"name": "Biryani", "calories": 350, "protein": 20, "carbs": 45, "fat": 12, "region": "Hyderabadi"},
        {"name": "Pulao", "calories": 250, "protein": 8, "carbs": 40, "fat": 6, "region": "North Indian"},
        {"name": "Jeera Rice", "calories": 200, "protein": 4, "carbs": 38, "fat": 5, "region": "North Indian"},
        
        # Breads
        {"name": "Naan", "calories": 320, "protein": 10, "carbs": 50, "fat": 8, "region": "North Indian"},
        {"name": "Roti", "calories": 120, "protein": 4, "carbs": 20, "fat": 2, "region": "North Indian"},
        {"name": "Paratha", "calories": 280, "protein": 7, "carbs": 35, "fat": 12, "region": "North Indian"},
        
        # South Indian
        {"name": "Dosa", "calories": 150, "protein": 4, "carbs": 25, "fat": 5, "region": "South Indian"},
        {"name": "Idli", "calories": 60, "protein": 3, "carbs": 12, "fat": 1, "region": "South Indian"},
        {"name": "Vada", "calories": 140, "protein": 5, "carbs": 18, "fat": 6, "region": "South Indian"},
        
        # Curries
        {"name": "Butter Chicken", "calories": 450, "protein": 30, "carbs": 15, "fat": 30, "region": "Punjabi"},
        {"name": "Palak Paneer", "calories": 300, "protein": 18, "carbs": 12, "fat": 20, "region": "North Indian"},
        {"name": "Chana Masala", "calories": 280, "protein": 15, "carbs": 35, "fat": 10, "region": "North Indian"},
        
        # Snacks
        {"name": "Samosa", "calories": 260, "protein": 5, "carbs": 30, "fat": 15, "region": "North Indian"},
        {"name": "Pakora", "calories": 180, "protein": 4, "carbs": 20, "fat": 10, "region": "North Indian"},
        {"name": "Bhel Puri", "calories": 200, "protein": 6, "carbs": 35, "fat": 5, "region": "Mumbai"},
        
        # Sweets
        {"name": "Gulab Jamun", "calories": 150, "protein": 3, "carbs": 25, "fat": 5, "region": "North Indian"},
        {"name": "Jalebi", "calories": 180, "protein": 2, "carbs": 40, "fat": 3, "region": "North Indian"},
        {"name": "Rasgulla", "calories": 120, "protein": 4, "carbs": 25, "fat": 1, "region": "Bengali"},
        
        # Street Food
        {"name": "Pani Puri", "calories": 100, "protein": 2, "carbs": 20, "fat": 2, "region": "Mumbai"},
        {"name": "Pav Bhaji", "calories": 350, "protein": 10, "carbs": 45, "fat": 15, "region": "Mumbai"},
        {"name": "Dahi Puri", "calories": 150, "protein": 5, "carbs": 25, "fat": 4, "region": "Mumbai"},
        
        # Healthy Options
        {"name": "Dal Khichdi", "calories": 200, "protein": 12, "carbs": 30, "fat": 5, "region": "Gujarati"},
        {"name": "Sambar Rice", "calories": 180, "protein": 8, "carbs": 30, "fat": 3, "region": "South Indian"},
        {"name": "Kitchari", "calories": 220, "protein": 10, "carbs": 35, "fat": 4, "region": "Ayurvedic"},
        
        # Additional Regional Specialties
        {"name": "Rogan Josh", "calories": 380, "protein": 25, "carbs": 10, "fat": 25, "region": "Kashmiri"},
        {"name": "Dhokla", "calories": 120, "protein": 6, "carbs": 20, "fat": 2, "region": "Gujarati"},
        {"name": "Misal Pav", "calories": 300, "protein": 15, "carbs": 40, "fat": 10, "region": "Maharashtrian"}
    ]
    
    try:
        if not DATASET_PATH.exists():
            os.makedirs(DATA_DIR, exist_ok=True)
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET,
                path=DATA_DIR,
                unzip=True
            )
        df = pd.read_csv(DATASET_PATH)
        # Combine with our expanded dataset
        expanded_df = pd.DataFrame(expanded_dataset)
        return pd.concat([df, expanded_df]).drop_duplicates(subset=['name'], keep='first')
    except Exception as e:
        logger.error(f"Failed to load Kaggle dataset: {e}")
        logger.info("Using expanded local dataset only")
        return pd.DataFrame(expanded_dataset)

# Load dataset at startup
food_df = load_food_dataset()

def analyze_food(image_path):
    """Analyze food using filename matching with dataset"""
    try:
        filename = secure_filename(os.path.basename(image_path))
        food_name = os.path.splitext(filename)[0].lower()
        
        # Find closest match in dataset (case insensitive)
        matched = food_df[food_df['name'].str.lower().str.contains(food_name, na=False)]
        if not matched.empty:
            return matched.iloc[0].to_dict()
        
        # Try fuzzy matching if exact match not found
        from fuzzywuzzy import process
        choices = food_df['name'].str.lower().tolist()
        best_match = process.extractOne(food_name, choices)
        if best_match and best_match[1] > 70:  # If similarity > 70%
            matched = food_df[food_df['name'].str.lower() == best_match[0]]
            if not matched.empty:
                return matched.iloc[0].to_dict()
        
        return None
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_food(filepath)
        if not result:
            return jsonify({
                "error": "Food not recognized",
                "suggestions": ["Try naming files like 'biryani.jpg'", 
                                "Supported foods: " + ", ".join(food_df['name'].sample(5).tolist())]
            }), 400
        
        health_condition = request.form.get('healthCondition', 'general')
        recommendations = get_recommendations(result['name'], health_condition)
        
        response = {
            "food": result['name'],
            "calories": result.get('calories', 0),
            "carbs": result.get('carbs', 0),
            "protein": result.get('protein', 0),
            "fat": result.get('fat', 0),
            "region": result.get('region', ''),
            "recommendations": recommendations,
            "image_path": f"/{filepath}"
        }
        
        # Add health score (0-100)
        health_score = min(100, max(0, 100 - (result.get('calories', 0)/5 + result.get('fat', 0)*2)))
        response['health_score'] = round(health_score)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
