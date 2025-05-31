import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# 1. Daten einlesen

file_path = "adult-2.csv"

df = pd.read_csv(file_path, delimiter=';')

# 2. Datensatz erkunden

print("\nErste 5 Zeilen des DataFrames:")

print(df.head())  # Erste Zeilen anzeigen

print("\nLetzte 5 Zeilen des DataFrames:")

print(df.tail())  # Letzte Zeilen anzeigen

# 3. Zusammenfassung des DataFrames

print("\nAllgemeine Informationen zum DataFrame:")

print(df.info())

print("\nStatistische Übersicht des DataFrames:")

print(df.describe())

# 4. Bedingte Auswahl (workclass ist ungleich ?)

df = df[df['workclass'] != '?']

print("\nZeilen mit workclass ist ungleich ?:")

print(df)

# 5. Nicht-numerische Werte in numerische)

print("\nNicht-numerische Werte in numerische:")

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes


print(df)

# 6. Korrelation

print("\nKorrelation:")

correlations = df[df.columns].corr(numeric_only=True)
print('Alle Korrelationen:')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum)
print('Schwächsten Korrelationen:')
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))

#7 Spalte mit geringster Korrelation entfernen

print('#7 Spalte mit geringster Korrelation entfernen:')

df = df.drop(['fnlwgt'], axis=1)

print(df)

#7 KNN Klassifikation von income

print("\nKNN Klassifikaion von income:")

"""Daten vorbereiten."""

# Was soll vorhergesagt werden? Spaltenname in Variable speichern.
col_name = 'income'

# Hier findet die Aufteilung in zwei Tabellen statt (Input=data und Output=col).
col = df[col_name]
data = df.drop([col_name], axis = 1)

"""KNN aufbauen"""

# Aus den zwei Tabellen vier Tabellen erzeugen
train_data, test_data, train_col, test_col = train_test_split(data,col, test_size=0.2, random_state=42)

# Aufbau KNN
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax)) # Ausgabeanzahl

# Konfiguration des Lernprozesses
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""Trainieren"""

cb_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.fit(train_data, train_col, epochs=100, validation_data=(test_data, test_col),callbacks=[cb_early])
