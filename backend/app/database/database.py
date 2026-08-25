import sqlite3
from pathlib import Path


# Location of the SQLite database
DATABASE_PATH = Path(__file__).resolve().parent.parent / "reconciliation.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reconciliation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            expected_amount REAL NOT NULL,
            actual_amount REAL NOT NULL,
            difference REAL NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            ai_explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

    

    connection.commit()
    connection.close()

def save_chat_message(session_id: str, role: str, message: str):
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO chat_messages (session_id, role, message)
            VALUES (?, ?, ?)
        """, (session_id, role, message))

        connection.commit()
        connection.close()
def get_chat_history(session_id: str):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, message
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))

    rows = cursor.fetchall()

    connection.close()

    return rows