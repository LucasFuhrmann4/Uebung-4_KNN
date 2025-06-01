import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# 1. Daten einlesen
file_path = "adult-2.csv"  # Dateipfad zur CSV-Datei
df = pd.read_csv(file_path, delimiter=';')  # Einlesen der CSV-Datei mit Semikolon als Trennzeichen

# 2. Datensatz erkunden
print("\nErste 5 Zeilen des DataFrames:")
print(df.head())  # Zeigt die ersten 5 Zeilen
print("\nLetzte 5 Zeilen des DataFrames:")
print(df.tail())  # Zeigt die letzten 5 Zeilen

# 3. Zusammenfassung des DataFrames
print("\nAllgemeine Informationen zum DataFrame:")
print(df.info())  # Überblick über Spalten, Datentypen, Nicht-Null-Werte
print("\nStatistische Übersicht des DataFrames:")
print(df.describe())  # Statistische Kenngrößen (nur numerisch)

# 4. Entferne Zeilen mit fehlenden oder ungültigen Werten in der Spalte 'workclass'
df = df[df['workclass'] != '?']
print("\nZeilen mit workclass ungleich '?':")
print(df)

# 5. Wandle alle nicht-numerischen Spalten in numerische Werte um
print("\nNicht-numerische Werte werden in numerische umgewandelt:")
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')  # String in Kategorie umwandeln
    df[col] = df[col].cat.codes  # Kategorien als Zahlen codieren
print(df)

# 6. Berechne Korrelationen zwischen allen numerischen Spalten
print("\nKorrelationen:")
correlations = df.corr(numeric_only=True)  # Korrelation nur für numerische Spalten
print("Alle Korrelationen (Summen der Absolutwerte je Spalte):")
print('-' * 30)
correlations_abs_sum = correlations.abs().sum()
print(correlations_abs_sum)
print("\nSpalten mit den geringsten Korrelationen:")
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))  # Die 5 Spalten mit der geringsten Gesamtkorrelation

# 7. Entferne eine Beispielspalte mit geringer Korrelation (hier: 'fnlwgt')
print("\nEntferne Spalte mit geringer Korrelation:")
df = df.drop(['fnlwgt'], axis=1)
print(df)

# 8. Aufbau eines einfachen Klassifikationsmodells für die Spalte 'income'
print("\nKNN-ähnliche Klassifikation von 'income':")

# Zielspalte festlegen (was soll vorhergesagt werden?)
col_name = 'income'
target = df[col_name]
features = df.drop([col_name], axis=1)

# Aufteilen in Trainings- und Testdaten (20% Testdaten)
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Aufbau des Modells mit TensorFlow Keras
# Hinweis: Hier wird kein klassisches KNN verwendet, sondern ein einfaches Feedforward-Netzwerk.
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(features.shape[1],)))  # Eingabeschicht
model.add(tf.keras.layers.Dense(40, activation='relu'))  # Erste versteckte Schicht
model.add(tf.keras.layers.Dense(80, activation='relu'))  # Zweite versteckte Schicht
model.add(tf.keras.layers.Dense(2, activation='softmax'))  # Ausgabeschicht (2 Klassen)

# Kompiliere das Modell mit Optimizer, Loss und Metriken
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # für Integer-Zielwerte
    metrics=['accuracy']
)

# Definiere EarlyStopping, um Overfitting zu vermeiden
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Trainiere das Modell
model.fit(
    train_features, train_target,
    epochs=100,
    validation_data=(test_features, test_target),
    callbacks=[early_stopping]
)

# Hinweis:
# - Dieses Modell kann leicht für andere Projekte angepasst werden, indem:
#   - die Datei oder Spaltennamen angepasst werden
#   - die Zielvariable geändert wird
#   - die Netzwerkarchitektur variiert wird
#   - zusätzliche Preprocessing-Schritte ergänzt werden