import yaml
import sqlite3
import os

print("--- Checking YAML ---")
try:
    with open('data/config.yaml', 'r') as f:
        data = yaml.safe_load(f)
        print("YAML Loaded successfully.")
        print(f"Cameras: {list(data.get('cameras', {}).keys())}")
except Exception as e:
    print(f"YAML Error: {e}")

print("\n--- Checking DB ---")
try:
    conn = sqlite3.connect('data/velovision.db')
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM events")
    count = cursor.fetchone()[0]
    print(f"Event Count: {count}")
    cursor.execute("SELECT count(*) FROM faces")
    f_count = cursor.fetchone()[0]
    print(f"Face Count: {f_count}")
    conn.close()
except Exception as e:
    print(f"DB Error: {e}")
