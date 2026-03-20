import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_price(features):
    features = np.array(features).reshape(1, -1)
    
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    
    return prediction[0]


