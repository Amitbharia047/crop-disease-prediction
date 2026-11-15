import random

# A mock ML model for crop recommendation to use during development.
# In production, this would use a trained Scikit-learn model (e.g., Random Forest)
# loaded via joblib or pickle.

CROPS = [
    "Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", 
    "Orange", "Apple", "Muskmelon", "Watermelon", "Grapes",
    "Mango", "Banana", "Pomegranate", "Lentil", "Blackgram",
    "Mungbean", "Mothbeans", "Pigeonpeas", "Kidneybeans", 
    "Chickpea", "Coffee"
]

def predict_crop(n: float, p: float, k: float, temperature: float, humidity: float, ph: float, rainfall: float):
    """
    Mock function to simulate a Machine Learning model predicting the best crop
    based on soil and environmental metrics.
    """
    # Simulate processing time
    import time
    time.sleep(0.5)
    
    # In a real scenario, we would scale the inputs and run `model.predict([[n, p, k, ...]])`
    
    # Let's add a little bit of pseudo-logic to make the mock slightly realistic
    if rainfall > 200 and humidity > 80:
        recommended = random.choice(["Rice", "Coconut", "Papaya"])
    elif temperature < 20 and ph < 6.5:
        recommended = random.choice(["Apple", "Grapes"])
    elif n > 80 and p > 40 and k > 40:
        recommended = random.choice(["Cotton", "Maize", "Banana"])
    else:
        recommended = random.choice(CROPS)
        
    return {
        "recommended_crop": recommended,
        "soil_health_score": round(random.uniform(6.0, 9.5), 1),
        "notes": f"The provided NPK values ({n}, {p}, {k}) and pH ({ph}) are suitable for {recommended}."
    }
