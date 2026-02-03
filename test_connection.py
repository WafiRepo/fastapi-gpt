#!/usr/bin/env python3
import mysql.connector
from mysql.connector import Error
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def test_mysql_connection():
    """Test MySQL connection"""
    print("\n[TEST] Testing MySQL Connection...")
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "sensor_data")
        )
        
        if connection.is_connected():
            print(f"[OK] MySQL Connection: SUCCESS")
            print(f"   Host: {os.getenv('DB_HOST', '127.0.0.1')}")
            print(f"   Database: {os.getenv('DB_NAME', 'sensor_data')}")
            print(f"   User: {os.getenv('DB_USER', 'root')}")
            
            # Test query
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"   Tables found: {len(tables)}")
            if tables:
                print(f"   Table names: {[t[0] for t in tables]}")
            
            # Check data counts
            cursor.execute("SELECT COUNT(*) FROM data_buffer")
            buffer_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM calculated_radius")
            radius_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM processed_images")
            images_count = cursor.fetchone()[0]
            
            print(f"   Data counts:")
            print(f"     - data_buffer: {buffer_count} records")
            print(f"     - calculated_radius: {radius_count} records")
            print(f"     - processed_images: {images_count} records")
            
            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"[FAIL] MySQL Connection: FAILED")
        print(f"   Error: {e}")
        print(f"   Make sure MySQL server is running!")
        return False
    except Exception as e:
        print(f"[FAIL] MySQL Connection: FAILED - {e}")
        return False

def test_fastapi_endpoint():
    """Test FastAPI endpoint (optional - only if server is running)"""
    print("\n[TEST] Testing FastAPI Endpoint...")
    try:
        # Try localhost first
        base_url = "http://localhost:8000"
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print(f"[OK] FastAPI Server: RUNNING at {base_url}")
            print(f"   API Docs: {base_url}/docs")
            return True
        else:
            print(f"[WARN] FastAPI Server: Responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[WARN] FastAPI Server: NOT RUNNING (this is OK if you haven't started it)")
        print(f"   To start: python main.py")
        return None  # Not an error, just not running
    except Exception as e:
        print(f"[WARN] FastAPI Endpoint: {e}")
        return None

def test_environment_variables():
    """Test if environment variables are loaded correctly"""
    print("\n[TEST] Testing Environment Variables...")
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "DB_HOST": os.getenv("DB_HOST", "127.0.0.1"),
        "DB_USER": os.getenv("DB_USER", "root"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", ""),
        "DB_NAME": os.getenv("DB_NAME", "sensor_data")
    }
    
    all_ok = True
    for var_name, var_value in required_vars.items():
        if var_value:
            if var_name == "OPENAI_API_KEY":
                print(f"[OK] {var_name}: SET (length: {len(var_value)})")
            elif var_name == "DB_PASSWORD":
                print(f"[OK] {var_name}: SET (length: {len(var_value)})")
            else:
                print(f"[OK] {var_name}: {var_value}")
        else:
            print(f"[FAIL] {var_name}: NOT SET")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("=" * 50)
    print("Testing G-Physic Backend Configuration")
    print("=" * 50)
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test MySQL connection
    mysql_ok = test_mysql_connection()
    
    # Test FastAPI endpoint (optional)
    api_ok = test_fastapi_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if env_ok:
        print("[OK] Environment Variables: OK")
    else:
        print("[FAIL] Environment Variables: MISSING")
    
    if mysql_ok:
        print("[OK] MySQL Database: CONNECTED")
    else:
        print("[FAIL] MySQL Database: FAILED (Server may not be running)")
    
    if api_ok is True:
        print("[OK] FastAPI Server: RUNNING")
    elif api_ok is None:
        print("[WARN] FastAPI Server: NOT RUNNING (optional)")
    else:
        print("[FAIL] FastAPI Server: ERROR")
    
    print("=" * 50)
    
    if env_ok and mysql_ok:
        print("[SUCCESS] Core components are ready!")
        if api_ok:
            print("[SUCCESS] Everything is working perfectly!")
        else:
            print("[INFO] Start FastAPI server with: python main.py")
    else:
        print("[WARN] Please fix the issues above before running the application")