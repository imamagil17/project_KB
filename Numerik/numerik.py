import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data simulasi pemain dan statistik mereka
data = {
    'Pemain': ['Messi', 'Ronaldo', 'Neymar', 'Boas', 'Gonzales', 'Benzema', 'M.Salah', 'Maradona'],
    'Shots': [5, 8, 7, 6, 9, 4, 3, 6],
    'Passes': [40, 50, 45, 60, 55, 30, 25, 35],
    'Dribbles': [3, 7, 5, 6, 8, 2, 1, 4],
    'Goals': [5, 3, 2, 1, 4, 0, 0, 1],
    'TopScorer': [1, 1, 0, 1, 1, 0, 0, 1]  # 1 jika pemain adalah pencetak gol terbanyak
}

# Membuat DataFrame dari data
df = pd.DataFrame(data)

# Memisahkan fitur dan label
X = df[['Shots', 'Passes', 'Dribbles', 'Goals']]  # Fitur: Shots, Passes, Dribbles, Goals
y = df['TopScorer']  # Label: TopScorer (0 = bukan pencetak gol terbanyak, 1 = pencetak gol terbanyak)

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()

# Menormalkan data latih dan uji, menjaga nama kolom
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Inisialisasi model K-NN dengan jumlah tetangga terdekat k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model dengan data latih
knn.fit(X_train_scaled, y_train)

# Memprediksi hasil pada data uji
y_pred = knn.predict(X_test_scaled)

# Evaluasi model
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred, zero_division=1))

# Menampilkan pemain yang merupakan pencetak gol terbanyak (gol lebih dari 0)
top_scorers = df[(df['TopScorer'] == 1) & (df['Goals'] > 0)]
print("\nPencetak Gol:")
for index, player in top_scorers.iterrows():
    print(f"{player['Pemain']} - {player['Goals']} Goals")

# Menentukan pencetak gol terbanyak berdasarkan jumlah gol terbanyak
top_score_player = df[df['Goals'] == df['Goals'].max()]

print("\nTop Score:")
for index, player in top_score_player.iterrows():
    print(f"{player['Pemain']} - {player['Goals']} Goals")
