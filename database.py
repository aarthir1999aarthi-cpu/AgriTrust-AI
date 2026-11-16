import sqlite3
import os


# ==================================================
# 1. DATABASE PATH
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "prediction_history.db"
)


# ==================================================
# 2. CREATE DATABASE
# ==================================================

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


# ==================================================
# 3. SAVE PREDICTION
# ==================================================

def save_prediction(crop, disease, confidence, reliability):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO prediction_history
        (crop, disease, confidence, reliability)

        VALUES (?, ?, ?, ?)
    """, (
        crop,
        disease,
        confidence,
        reliability
    ))

    connection.commit()

    connection.close()


# ==================================================
# 4. GET PREDICTION HISTORY
# ==================================================

def get_prediction_history():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM prediction_history
        ORDER BY id DESC
    """)

    records = [
        dict(row)
        for row in cursor.fetchall()
    ]

    connection.close()

    return records


# ==================================================
# 5. INITIALIZE DATABASE
# ==================================================

create_database()

print("Prediction History Database Ready!")