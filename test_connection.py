#!/usr/bin/env python3
import mysql.connector
import requests

def test_mysql_connection():
    """Test MySQL connection - HANYA SEKALI"""
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",  # Localhost
            user="fastapi_user",
            password="SecurePass123!",
            database="sensor_data"
        )
        print("✅ MySQL Connection: SUCCESS")
        connection.close()
        return True
    except Exception as e:
        print(f"❌ MySQL Connection: FAILED - {e}")
        return False

def test_fastapi_endpoint():
    """Test FastAPI endpoint"""
    try:
        response = requests.post(
            "http://148.230.96.39:8000/add_buffer_data/",
            json={"test": "data"},
            timeout=10
        )
        print(f"✅ FastAPI Endpoint: SUCCESS - Status: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ FastAPI Endpoint: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Connections...")
    print("-" * 40)
    
    # HANYA PANGGIL SEKALI!
    mysql_ok = test_mysql_connection()  # Call 1
    api_ok = test_fastapi_endpoint()    # Call 1
    
    print("-" * 40)
    if mysql_ok and api_ok:
        print("🎉 All connections working!")
    else:
        print("⚠️  Some connections failed!")