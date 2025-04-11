import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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