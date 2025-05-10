import mysql.connector
from datetime import datetime
from werkzeug.security import generate_password_hash
import json

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="arjun@2004",
        database="health_db"
    )

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            age INT,
            gender VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create health_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            height FLOAT,
            weight FLOAT,
            bmi FLOAT,
            blood_pressure VARCHAR(20),
            heart_rate INT,
            physical_activity_level FLOAT,
            sleep_hours FLOAT,
            smoking_status INT,
            alcohol_consumption INT,
            existing_conditions TEXT,
            family_history TEXT,
            medications TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create health_data table with expanded fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            age FLOAT,
            weight FLOAT,
            height FLOAT,
            bmi FLOAT,
            blood_pressure VARCHAR(20),
            heart_rate INT,
            glucose_level FLOAT,
            cholesterol FLOAT,
            hdl_cholesterol FLOAT,
            ldl_cholesterol FLOAT,
            triglycerides FLOAT,
            smoking_status INT,
            alcohol_consumption INT,
            physical_activity_level FLOAT,
            sleep_hours FLOAT,
            stress_level INT,
            diet_type VARCHAR(50),
            water_intake FLOAT,
            family_history TEXT,
            existing_conditions TEXT,
            medications TEXT,
            allergies TEXT,
            last_checkup DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create medical_reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_path VARCHAR(255),
            analysis_results JSON,
            key_findings JSON,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create health_predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            prediction_type VARCHAR(50),
            risk_score FLOAT,
            confidence_level FLOAT,
            timeline_prediction TEXT,
            recommendations TEXT,
            prevention_steps TEXT,
            lifestyle_changes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create health_timeline table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_timeline (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            event_type VARCHAR(50),
            event_description TEXT,
            severity_level INT,
            predicted_date DATE,
            probability FLOAT,
            preventive_actions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Add pdf_analysis table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_analysis (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            file_name VARCHAR(255),
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metrics JSON,
            findings TEXT,
            diagnoses TEXT,
            recommendations TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

# User Management Functions
def add_user(name, email, password, age=None, gender=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Hash the password before storing
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (name, email, password, age, gender) VALUES (%s, %s, %s, %s, %s)",
            (name, email, hashed_password, age, gender)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_user_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def update_user_profile(user_id, data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = """
            UPDATE users 
            SET name = %s, age = %s, gender = %s
            WHERE id = %s
        """
        cursor.execute(query, (data['name'], data['age'], data['gender'], user_id))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

# Health Data Management Functions
def save_health_data(user_id, health_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO health_data 
            (user_id, age, weight, height, bmi, blood_pressure, glucose_level, 
            cholesterol, smoking_status, alcohol_consumption, physical_activity_level,
            sleep_hours, stress_level, diet_type, water_intake,
            family_history, existing_conditions, medications, allergies, last_checkup)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id, health_data['age'], health_data['weight'], health_data['height'],
            health_data['bmi'], health_data['blood_pressure'], health_data['glucose_level'],
            health_data['cholesterol'], health_data['smoking_status'],
            health_data['alcohol_consumption'], health_data['physical_activity_level'],
            health_data['sleep_hours'], health_data['stress_level'], 
            health_data['diet_type'], health_data['water_intake'],
            health_data['family_history'], health_data['existing_conditions'],
            health_data['medications'], health_data['allergies'], 
            health_data['last_checkup']
        ))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_health_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT * FROM health_data 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (user_id,))
        health_data = cursor.fetchone()
        
        if health_data:
            # Convert None values to appropriate defaults
            health_data = {k: (v if v is not None else 0) for k, v in health_data.items()}
            return health_data
        return None
    except Exception as e:
        print(f"Error getting health data: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

# Medical Reports Management
def save_medical_report(user_id, report_type, file_name, file_path, analysis_results, doctor_comments=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO medical_reports 
            (user_id, report_type, file_name, file_path, analysis_results, doctor_comments, report_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, report_type, file_name, file_path, analysis_results, doctor_comments, datetime.now().date()))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_medical_reports(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM medical_reports WHERE user_id = %s ORDER BY uploaded_at DESC", (user_id,))
    reports = cursor.fetchall()
    cursor.close()
    conn.close()
    return reports

# Health Predictions and Timeline Management
def save_health_prediction(user_id, prediction_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO health_predictions 
            (user_id, prediction_type, risk_score, confidence_level, 
            timeline_prediction, recommendations, prevention_steps, lifestyle_changes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id, prediction_data['prediction_type'], prediction_data['risk_score'],
            prediction_data['confidence_level'], prediction_data['timeline_prediction'],
            prediction_data['recommendations'], prediction_data['prevention_steps'],
            prediction_data['lifestyle_changes']
        ))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def save_timeline_event(user_id, timeline_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO health_timeline 
            (user_id, event_type, event_description, severity_level, 
            predicted_date, probability, preventive_actions)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id, timeline_data['event_type'], timeline_data['event_description'],
            timeline_data['severity_level'], timeline_data['predicted_date'],
            timeline_data['probability'], timeline_data['preventive_actions']
        ))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_user_predictions(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM health_predictions WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    predictions = cursor.fetchall()
    cursor.close()
    conn.close()
    return predictions

def get_user_timeline(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM health_timeline WHERE user_id = %s ORDER BY predicted_date ASC", (user_id,))
    timeline = cursor.fetchall()
    cursor.close()
    conn.close()
    return timeline

def save_pdf_analysis(user_id, file_name, analysis_results):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO pdf_analysis 
            (user_id, file_name, metrics, findings, diagnoses, recommendations)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            file_name,
            json.dumps(analysis_results['analysis']['metrics']),
            json.dumps(analysis_results['analysis']['key_findings']),
            json.dumps(analysis_results['analysis']['diagnoses']),
            json.dumps(analysis_results['analysis']['recommendations'])
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving PDF analysis: {str(e)}")
        return False
    finally:
        cursor.close()
        conn.close()

def update_health_data_from_pdf(user_id, metrics):
    """
    Update user's health data with metrics extracted from PDF
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # First, get existing health data
        cursor.execute("SELECT * FROM health_data WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,))
        existing_data = cursor.fetchone()
        
        # Prepare update data
        update_data = {}
        if existing_data:
            # Convert existing data to dictionary
            columns = [desc[0] for desc in cursor.description]
            existing_data_dict = dict(zip(columns, existing_data))
            update_data = existing_data_dict.copy()
        
        # Update with new metrics
        if metrics.get('blood_pressure'):
            update_data['blood_pressure'] = metrics['blood_pressure']
        if metrics.get('heart_rate'):
            update_data['heart_rate'] = int(metrics['heart_rate'])
        if metrics.get('glucose'):
            update_data['glucose_level'] = float(metrics['glucose'])
        if metrics.get('cholesterol'):
            update_data['cholesterol'] = float(metrics['cholesterol'])
        if metrics.get('bmi'):
            update_data['bmi'] = float(metrics['bmi'])
        if metrics.get('weight'):
            update_data['weight'] = float(metrics['weight'])
        if metrics.get('height'):
            update_data['height'] = float(metrics['height'])
            
        # If there's existing data, update it
        if existing_data:
            set_clause = ", ".join([f"{k} = %s" for k in update_data.keys()])
            values = list(update_data.values())
            values.append(user_id)  # for WHERE clause
            
            cursor.execute(f"""
                UPDATE health_data 
                SET {set_clause}
                WHERE user_id = %s
            """, tuple(values))
        else:
            # Insert new record
            columns = ", ".join(update_data.keys())
            placeholders = ", ".join(["%s"] * len(update_data))
            cursor.execute(f"""
                INSERT INTO health_data (user_id, {columns})
                VALUES (%s, {placeholders})
            """, (user_id, *update_data.values()))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error updating health data from PDF: {str(e)}")
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()