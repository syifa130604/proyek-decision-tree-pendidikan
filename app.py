from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model yang sudah dilatih
model = joblib.load('model_decision_tree.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data input dari form HTML
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Melakukan prediksi dengan model Decision Tree
        prediction = model.predict(final_features)
        
        # Menerjemahkan hasil prediksi angka ke teks
        if prediction[0] == 1:
            result = "Status: TINGGI (Rawan Kerusakan Berat)"
        else:
            result = "Status: RENDAH (Aman / Terkendali)"
            
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Terjadi kesalahan input: {e}")

if __name__ == "__main__":
    app.run(debug=True)