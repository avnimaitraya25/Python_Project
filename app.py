from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'RBA_Final_dataset.csv'
df = pd.read_csv(file_path, low_memory=False, dtype=str)

# Fill missing values and drop unneeded columns
df = df.drop(columns=['index', 'Login Timestamp', 'IP Address', 'User Agent String'])
df['Region'] = df['Region'].fillna('Unknown')
df['City'] = df['City'].fillna('Unknown')
df['Round-Trip Time [ms]'] = pd.to_numeric(df['Round-Trip Time [ms]'], errors='coerce').fillna(0)

# Label encode categorical variables
label_encoders = {}
categorical_cols = ['User ID', 'Country', 'Region', 'City', 'ASN',
                    'Browser Name and Version', 'OS Name and Version', 'Device Type']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target encoding
df['Is Attack IP'] = df['Is Attack IP'].astype(bool).astype(int)

# Features and target
X = df.drop(columns=['Is Attack IP', 'Is Account Takeover', 'Login Successful'])
y = df['Is Attack IP']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Accuracy: {accuracy:.3f}")

# Save the model
model_file = 'rba_model.pkl'
joblib.dump((model, label_encoders), model_file)

# Web Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_data = {
            'User ID': data['userId'],
            'Round-Trip Time [ms]': float(data['rtt']),
            'Country': data['country'],
            'Region': data['region'] or 'Unknown',
            'City': data['city'] or 'Unknown',
            'ASN': data['asn'],
            'Browser Name and Version': data['browser'],
            'OS Name and Version': data['os'],
            'Device Type': data['deviceType']
        }

        # Encode the inputs
        encoded_input = []
        for col in X.columns:
            if col in input_data:
                val = input_data[col]
                if col in label_encoders:
                    val = label_encoders[col].transform([str(val)])[0]
                encoded_input.append(val)
            else:
                encoded_input.append(0)

        # Predict
        model, _ = joblib.load(model_file)
        prediction = model.predict([encoded_input])[0]

        risk_level = 'High Risk' if prediction == 1 else 'Low Risk'
        return jsonify({'risk': risk_level})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

