import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# READING DATA
# Formatting adjustment for Bolivian regional CSV
df = pd.read_csv('datos_sensores.csv', sep=';', decimal=',')

# PREPARATION
X = df[['temp', 'hum', 'co2', 'smoke']]
y = df['fire_label']

# TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL CREATION
# Using Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# TRAINING
print("Training SIDTMIS brain...")
modelo.fit(X_train, y_train)

# EVALUATION
predicciones = modelo.predict(X_test)
print("\n--- RESULTS ---")
print(classification_report(y_test, predicciones))

# MATRIX
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, predicciones))
