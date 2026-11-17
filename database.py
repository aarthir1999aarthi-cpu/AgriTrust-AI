import sqlite3
import os

# DATABASE PATH

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_DIR = os.path.join(BASE_DIR, "database")

# Automatically create database folder
os.makedirs(DATABASE_DIR, exist_ok=True)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "prediction_history.db"
)


# CREATE DATABASE

def create_database():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            reliability TEXT NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


# SAVE PREDICTION

def save_prediction(crop, disease, confidence, reliability):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction_history
        (crop, disease, confidence, reliability)
        VALUES (?, ?, ?, ?)
    """, (crop, disease, confidence, reliability))

    connection.commit()
    connection.close()


# GET PREDICTION HISTORY

def get_prediction_history():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
    """)

    records = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return records


# INITIALIZE DATABASE

create_database()

print("Prediction History Database Ready!")
```
