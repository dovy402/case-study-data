import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Membuat Dummy Dataset untuk Simulasi (atau load dari file CSV)
np.random.seed(42)
data = {
    'Usia': np.random.randint(18, 65, 100),
    'Durasi_Aktivitas_Jam': np.random.uniform(0.5, 10.0, 100),
    'Jumlah_Transaksi': np.random.randint(1, 20, 100),
    'Status_Loyalitas': np.random.choice([0, 1], size=100, p=[0.6, 0.4]) # 0: Reguler, 1: VIP
}
df = pd.DataFrame(data)

print("--- 5 Baris Pertama Dataset ---")
print(df.head(), "\n")

# 2. Data Analysis / EDA Sederhana
print("--- Ringkasan Statistik Data ---")
print(df.describe(), "\n")

print("--- Rata-rata Durasi Aktivitas berdasarkan Status Loyalitas ---")
print(df.groupby('Status_Loyalitas')['Durasi_Aktivitas_Jam'].mean(), "\n")

# 3. Implementasi AI (Machine Learning untuk Klasifikasi)
X = df[['Usia', 'Durasi_Aktivitas_Jam', 'Jumlah_Transaksi']]
y = df['Status_Loyalitas']

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)
print("--- Evaluasi Model AI ---")
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))