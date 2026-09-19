import sys
import os
import json
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model

def predict(text):
    if not text or not text.strip():
        return {
            "predicted_category": "Other",
            "confidence": 0.5,
            "all_probabilities": {}
        }
    
    model = get_model()
    text_clean = text.strip()
    
    # Predict category
    predicted_cat = model.predict([text_clean])[0]
    
    # Predict probabilities
    probs = model.predict_proba([text_clean])[0]
    classes = model.classes_
    
    cat_probs = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
    confidence = round(float(np_max := max(probs)), 4)
    
    return {
        "predicted_category": predicted_cat,
        "confidence": confidence,
        "all_probabilities": cat_probs
    }

if __name__ == "__main__":
    # Support CLI arguments or stdin
    import argparse
    parser = argparse.ArgumentParser(description="Predict expense category from text")
    parser.add_argument("--text", type=str, help="Text to classify (vendor, description, items)")
    args = parser.parse_args()
    
    input_text = ""
    if args.text:
        input_text = args.text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read()
    
    try:
        result = predict(input_text)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e), "predicted_category": "Other", "confidence": 0.0}))
        sys.exit(1)
