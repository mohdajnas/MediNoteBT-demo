from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import os
import google.generativeai as genai
from datetime import datetime
import json
from dotenv import load_dotenv
# Add these imports at the top
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
import secrets
from urllib.parse import urljoin
import json
from datetime import datetime


load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class DischargeSummaryGenerator:
    def __init__(self, gemini_api_key, csv_file_path):
        """
        Initialize the discharge summary generator
        
        Args:
            gemini_api_key: Your Google Gemini API key
            csv_file_path: Path to the CSV file containing patient data
        """
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.csv_file_path = csv_file_path
        self.patient_df = None
        self.load_patient_data()
    
    def load_patient_data(self):
        """Load patient data from CSV file with memory optimization"""
        try:
            # Define data types for memory optimization
            dtype_dict = {
                'Patient ID': 'string',
                'Name': 'string',
                'Age': 'int16',
                'Gender': 'category',
                'Primary Diagnosis': 'string',
                'Secondary Diagnosis 1': 'string',
                'Secondary Diagnosis 2': 'string',
                'Procedure': 'string',
                'Admission Date': 'string',
                'Course in Hospital': 'string',
                'Medications': 'string',
                'Follow-up Instructions': 'string',
                'Rehabilitation Plan': 'string',
                'Other Instructions': 'string'
            }
            
            # Try reading with different encodings and optimized data types
            try:
                self.patient_df = pd.read_csv(
                    self.csv_file_path, 
                    encoding='utf-8',
                    dtype=dtype_dict,
                    low_memory=False
                )
            except:
                try:
                    self.patient_df = pd.read_csv(
                        self.csv_file_path, 
                        encoding='latin-1',
                        dtype=dtype_dict,
                        low_memory=False
                    )
                except:
                    self.patient_df = pd.read_csv(
                        self.csv_file_path, 
                        encoding='iso-8859-1',
                        dtype=dtype_dict,
                        low_memory=False
                    )
            
            # Strip whitespace from column names
            self.patient_df.columns = self.patient_df.columns.str.strip()
            
            # Strip whitespace from Patient ID values
            if 'Patient ID' in self.patient_df.columns:
                self.patient_df['Patient ID'] = self.patient_df['Patient ID'].str.strip()
            
            # Optimize memory usage
            self.patient_df = self._optimize_dataframe(self.patient_df)
            
            # Calculate memory usage
            memory_usage = self.patient_df.memory_usage(deep=True).sum() / 1024**2
            print(f"Successfully loaded {len(self.patient_df)} patient records")
            print(f"Memory usage: {memory_usage:.2f} MB")
            print(f"CSV columns: {list(self.patient_df.columns)}")
            if 'Patient ID' in self.patient_df.columns:
                print(f"Sample Patient IDs (first 5): {self.patient_df['Patient ID'].head().tolist()}")
            else:
                print(f"WARNING: 'Patient ID' column not found!")
                print(f"Available columns: {list(self.patient_df.columns)}")
        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_file_path}")
            # Create sample data if file doesn't exist
            self.patient_df = self.create_sample_data()
            print("Using sample data instead")
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            import traceback
            traceback.print_exc()
            self.patient_df = self.create_sample_data()
            print("Using sample data instead")
    
    def _optimize_dataframe(self, df):
        """Optimize dataframe memory usage"""
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['Patient ID', 'Name', 'Course in Hospital', 'Medications', 
                          'Follow-up Instructions', 'Rehabilitation Plan', 'Other Instructions']:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
        
        return df
    
    def create_sample_data(self):
        """Create sample patient data for demonstration"""
        sample_data = [
            {
                'Patient ID': 'N4894',
                'Name': 'John Smith',
                'Age': 45,
                'Gender': 'Male',
                'Primary Diagnosis': 'Acute Myocardial Infarction',
                'Secondary Diagnosis 1': 'Hypertension',
                'Secondary Diagnosis 2': 'Type 2 Diabetes',
                'Procedure': 'Percutaneous Coronary Intervention',
                'Admission Date': '2025-01-15',
                'Course in Hospital': 'Patient presented with chest pain and was diagnosed with STEMI. Successfully underwent PCI with stent placement.',
                'Medications': 'Aspirin 81mg daily, Metoprolol 50mg BID, Lisinopril 10mg daily',
                'Follow-up Instructions': 'Cardiology follow-up in 2 weeks, Primary care in 1 week',
                'Rehabilitation Plan': 'Cardiac rehabilitation program enrollment',
                'Other Instructions': 'No heavy lifting for 2 weeks, smoking cessation counseling'
            },
            {
                'Patient ID': 'N4895',
                'Name': 'Sarah Johnson',
                'Age': 32,
                'Gender': 'Female',
                'Primary Diagnosis': 'Pneumonia',
                'Secondary Diagnosis 1': 'Asthma',
                'Secondary Diagnosis 2': 'None',
                'Procedure': 'None',
                'Admission Date': '2025-01-10',
                'Course in Hospital': 'Patient treated with IV antibiotics for community-acquired pneumonia with good response.',
                'Medications': 'Azithromycin 500mg daily, Albuterol inhaler PRN',
                'Follow-up Instructions': 'Primary care follow-up in 1 week',
                'Rehabilitation Plan': 'None',
                'Other Instructions': 'Continue prescribed medications, return if symptoms worsen'
            }
        ]
        return pd.DataFrame(sample_data)
    
    def fetch_patient_record(self, patient_id):
        """
        Fetch patient record by unique patient ID
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            dict: Patient record or None if not found
        """
        if self.patient_df is None or self.patient_df.empty:
            print(f"No patient data loaded")
            return None
        
        # Strip whitespace from search term
        patient_id = str(patient_id).strip()
        
        print(f"Searching for patient ID: '{patient_id}'")
        print(f"Available IDs: {self.patient_df['Patient ID'].tolist()}")
        
        patient_record = self.patient_df[self.patient_df['Patient ID'] == patient_id]
        
        if patient_record.empty:
            print(f"Patient ID '{patient_id}' not found in database")
            return None
        
        print(f"Found patient: {patient_record.iloc[0]['Name']}")
        return patient_record.iloc[0].to_dict()
    
    def format_patient_data_for_llm(self, patient_record):
        """
        Format patient data into a structured prompt for the LLM
        
        Args:
            patient_record: Dictionary containing patient information
            
        Returns:
            str: Formatted prompt for LLM
        """
        prompt = f"""
        You are a medical professional tasked with generating a comprehensive discharge summary. 
        Use the following patient data to create a well-structured discharge summary in the exact format provided below.

        Patient Data:
        - Patient ID: {patient_record.get('Patient ID', 'N/A')}
        - Name: {patient_record.get('Name', 'N/A')}
        - Age: {patient_record.get('Age', 'N/A')}
        - Gender: {patient_record.get('Gender', 'N/A')}
        - Primary Diagnosis: {patient_record.get('Primary Diagnosis', 'N/A')}
        - Secondary Diagnosis 1: {patient_record.get('Secondary Diagnosis 1', 'N/A')}
        - Secondary Diagnosis 2: {patient_record.get('Secondary Diagnosis 2', 'N/A')}
        - Procedure: {patient_record.get('Procedure', 'None')}
        - Admission Date: {patient_record.get('Admission Date', 'N/A')}
        - Course in Hospital: {patient_record.get('Course in Hospital', 'N/A')}
        - Medications: {patient_record.get('Medications', 'N/A')}
        - Follow-up Instructions: {patient_record.get('Follow-up Instructions', 'N/A')}
        - Rehabilitation Plan: {patient_record.get('Rehabilitation Plan', 'None')}
        - Other Instructions: {patient_record.get('Other Instructions', 'None')}

        Please generate a discharge summary following this exact format:

        **Patient Information**:
        - Name: [Patient Name]
        - Age: [Age]
        - Gender: [Gender]
        - Patient ID: [Patient ID]

        **Diagnosis**:
        - Primary Diagnosis: [Primary Diagnosis]
        - Secondary Diagnosis: [Secondary Diagnosis 1]
        - Secondary Diagnosis: [Secondary Diagnosis 2 if available]

        **Procedure**: [Procedure or "None"]

        **Hospital Course**:
        Admission Date: [Format as DD/MM/YYYY]
        Course in Hospital: [Detailed course description]

        **Discharge Plan**:
        - Medications: [Medication details]
        - Follow-up Instructions: [Follow-up instructions]
        - Rehabilitation Plan: [Rehabilitation plan or "None"]
        - Other Instructions: [Other instructions or "None"]

        Generate only the discharge summary without any additional commentary.
        """
        return prompt
    
    def generate_summary_with_gemini(self, patient_record):
        """
        Generate discharge summary using Gemini API with memory optimization
        
        Args:
            patient_record: Dictionary containing patient information
            
        Returns:
            str: Generated discharge summary
        """
        try:
            prompt = self.format_patient_data_for_llm(patient_record)
            
            # Configure generation parameters with token limits for memory efficiency
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent medical documentation
                max_output_tokens=800,  # Reduced from 1024 for memory optimization
                top_p=0.9,
                top_k=40
            )
            
            # Generate the content
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Clear the prompt from memory immediately after use
            del prompt
            
            return response.text
            
        except Exception as e:
            return f"Error generating summary with Gemini API: {str(e)}"
    
    def generate_discharge_summary(self, patient_id):
        """
        Main method to generate discharge summary for a given patient ID
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            str: Generated discharge summary or error message
        """
        # Step 1: Fetch patient record from CSV
        patient_record = self.fetch_patient_record(patient_id)
        
        if patient_record is None:
            return f"Patient with ID '{patient_id}' not found in the database."
        
        # Step 2: Generate summary using Gemini LLM
        discharge_summary = self.generate_summary_with_gemini(patient_record)
        
        return discharge_summary

# Initialize the generator
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CSV_FILE_PATH = "patient_data.csv"



if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")
    print("Please set your GEMINI_API_KEY environment variable")

generator = DischargeSummaryGenerator(GEMINI_API_KEY, CSV_FILE_PATH)
pending_summaries = {}  # Store pending summaries with tokens
approved_summaries = {}  # Store approved/modified summaries

# Memory optimization: Limit pending and approved summaries
MAX_PENDING_SUMMARIES = 100
MAX_APPROVED_SUMMARIES = 500

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
BASE_URL = os.getenv('BASE_URL', 'https://your-app-name.onrender.com')


doctors_list = [
    {"name": "Dr. Aleem Ahammed", "email": "aleemahamedf@gmail.com", "specialty": "Cardiology"},
    {"name": "Dr. Ajnas", "email": "mohdajnas@yahoo.com", "specialty": "Internal Medicine"},
    {"name": "Dr. Eleem", "email": "info@boehmtech.co", "specialty": "Pulmonology"},
    {"name": "Dr. Wilson", "email": "dr.wilson@hospital.com", "specialty": "General Medicine"},
    {"name": "Dr. Davis", "email": "dr.davis@hospital.com", "specialty": "Oncology"}
]

def cleanup_old_summaries():
    """Remove old summaries to prevent memory buildup"""
    current_time = datetime.now()
    
    # Clean up pending summaries older than 24 hours
    expired_tokens = []
    for token, data in list(pending_summaries.items()):
        created_at = datetime.fromisoformat(data['created_at'])
        if (current_time - created_at).total_seconds() > 86400:  # 24 hours
            expired_tokens.append(token)
    
    for token in expired_tokens:
        del pending_summaries[token]
    
    # Limit pending summaries
    if len(pending_summaries) > MAX_PENDING_SUMMARIES:
        sorted_keys = sorted(pending_summaries.keys(), 
                           key=lambda x: pending_summaries[x]['created_at'])
        for key in sorted_keys[:len(pending_summaries) - MAX_PENDING_SUMMARIES]:
            del pending_summaries[key]
    
    # Limit approved summaries
    if len(approved_summaries) > MAX_APPROVED_SUMMARIES:
        sorted_keys = sorted(approved_summaries.keys(), 
                           key=lambda x: approved_summaries[x]['approved_at'])
        for key in sorted_keys[:len(approved_summaries) - MAX_APPROVED_SUMMARIES]:
            del approved_summaries[key]
    
    if expired_tokens:
        print(f"Cleaned up {len(expired_tokens)} expired summaries")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """
    API endpoint to fetch patient data by ID
    """
    try:
        patient_record = generator.fetch_patient_record(patient_id)
        
        if patient_record is None:
            return jsonify({
                'success': False,
                'message': f"Patient with ID '{patient_id}' not found"
            }), 404
        
        # Convert any NaN values to None for JSON serialization
        for key, value in patient_record.items():
            if pd.isna(value):
                patient_record[key] = None
        
        return jsonify({
            'success': True,
            'data': patient_record
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error fetching patient data: {str(e)}"
        }), 500

@app.route('/api/generate-summary', methods=['POST'])
def generate_summary():
    """
    API endpoint to generate discharge summary
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        
        if not patient_id:
            return jsonify({
                'success': False,
                'message': 'Patient ID is required'
            }), 400
        
        # Check if Gemini API key is available
        if not GEMINI_API_KEY:
            # Return mock summary for demonstration
            patient_record = generator.fetch_patient_record(patient_id)
            if patient_record is None:
                return jsonify({
                    'success': False,
                    'message': f"Patient with ID '{patient_id}' not found"
                }), 404
            
            mock_summary = generate_mock_summary(patient_record)
            return jsonify({
                'success': True,
                'summary': mock_summary
            })
        
        # Generate actual summary using Gemini
        summary = generator.generate_discharge_summary(patient_id)
        
        if summary.startswith("Patient with ID"):
            return jsonify({
                'success': False,
                'message': summary
            }), 404
        
        if summary.startswith("Error generating summary"):
            return jsonify({
                'success': False,
                'message': summary
            }), 500
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error generating summary: {str(e)}"
        }), 500

def generate_mock_summary(patient_record):
    """Generate a mock summary for demonstration purposes"""
    admission_date = patient_record.get('Admission Date', '')
    if admission_date:
        try:
            formatted_date = datetime.strptime(admission_date, '%Y-%m-%d').strftime('%d/%m/%Y')
        except:
            formatted_date = admission_date
    else:
        formatted_date = 'N/A'
    
    secondary_diag_2 = patient_record.get('Secondary Diagnosis 2', '')
    secondary_diag_2_line = f"\n- Secondary Diagnosis: {secondary_diag_2}" if secondary_diag_2 and secondary_diag_2.lower() != 'none' else ""
    
    return f"""**Patient Information**:
- Name: {patient_record.get('Name', 'N/A')}
- Age: {patient_record.get('Age', 'N/A')}
- Gender: {patient_record.get('Gender', 'N/A')}
- Patient ID: {patient_record.get('Patient ID', 'N/A')}

**Diagnosis**:
- Primary Diagnosis: {patient_record.get('Primary Diagnosis', 'N/A')}
- Secondary Diagnosis: {patient_record.get('Secondary Diagnosis 1', 'N/A')}{secondary_diag_2_line}

**Procedure**: {patient_record.get('Procedure', 'None')}

**Hospital Course**:
Admission Date: {formatted_date}
Course in Hospital: {patient_record.get('Course in Hospital', 'N/A')}

**Discharge Plan**:
- Medications: {patient_record.get('Medications', 'N/A')}
- Follow-up Instructions: {patient_record.get('Follow-up Instructions', 'N/A')}
- Rehabilitation Plan: {patient_record.get('Rehabilitation Plan', 'None')}
- Other Instructions: {patient_record.get('Other Instructions', 'None')}"""

@app.route('/api/save-summary', methods=['POST'])
def save_summary():
    """
    API endpoint to save generated summary to file
    """
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        summary = data.get('summary')
        
        if not patient_id or not summary:
            return jsonify({
                'success': False,
                'message': 'Patient ID and summary are required'
            }), 400
        
        # Create output directory if it doesn't exist
        output_dir = "discharge_summaries"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_discharge_summary_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return jsonify({
            'success': True,
            'message': f'Summary saved to {filepath}',
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error saving summary: {str(e)}"
        }), 500
    
@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    """
    API endpoint to get list of doctors
    """
    try:
        return jsonify({
            'success': True,
            'doctors': doctors_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error fetching doctors: {str(e)}"
        }), 500 
    
@app.route('/api/send-for-approval', methods=['POST'])
def send_for_approval():
    """Send summary to doctor for approval via email"""
    try:
        # Clean up old summaries before adding new ones
        cleanup_old_summaries()
        
        data = request.get_json()
        patient_id = data.get('patient_id')
        summary = data.get('summary')
        doctor_email = data.get('doctor_email')
        
        if not all([patient_id, summary, doctor_email]):
            return jsonify({
                'success': False,
                'message': 'Patient ID, summary, and doctor email are required'
            }), 400
        
        # Generate unique approval token
        approval_token = secrets.token_urlsafe(32)
        
        # Store pending summary with minimal data
        pending_summaries[approval_token] = {
            'patient_id': patient_id,
            'summary': summary,
            'doctor_email': doctor_email,
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Send email to doctor
        approval_link = f"{BASE_URL}/approve/{approval_token}"
        
        if send_approval_email(doctor_email, patient_id, summary, approval_link):
            return jsonify({
                'success': True,
                'message': 'Approval email sent successfully',
                'token': approval_token
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send approval email'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error sending approval: {str(e)}"
        }), 500

@app.route('/approve/<token>')
def approve_page(token):
    """Lightweight approval page for doctors"""
    if token not in pending_summaries:
        return render_template('error.html', 
                             message="Invalid or expired approval link"), 404
    
    summary_data = pending_summaries[token]
    return render_template('approve.html', 
                         token=token, 
                         summary_data=summary_data)

@app.route('/api/approve-summary', methods=['POST'])
def approve_summary():
    """Handle doctor's approval with optional edits"""
    try:
        data = request.get_json()
        token = data.get('token')
        action = data.get('action')  # only 'approve' now
        modified_summary = data.get('modified_summary', '')
        
        if token not in pending_summaries:
            return jsonify({
                'success': False,
                'message': 'Invalid or expired approval token'
            }), 404
        
        summary_data = pending_summaries[token]
        
        if action == 'approve':
            # Determine if doctor made edits
            original_summary = summary_data['summary'].strip()
            edited_summary = modified_summary.strip()
            doctor_made_edits = edited_summary != original_summary
            
            # Store only essential data in approved summaries
            approved_summaries[token] = {
                'patient_id': summary_data['patient_id'],
                'final_summary': edited_summary if doctor_made_edits else original_summary,
                'status': 'approved',
                'approved_at': datetime.now().isoformat(),
                'modified_by_doctor': doctor_made_edits
            }
            
            # Update patient record
            final_summary = edited_summary if doctor_made_edits else original_summary
            update_patient_record(summary_data['patient_id'], final_summary)
            
            # Remove from pending to free memory
            del pending_summaries[token]
            
            # Clean up old summaries
            cleanup_old_summaries()
            
            return jsonify({
                'success': True,
                'message': 'Summary approved successfully',
                'status': 'approved',
                'final_summary': final_summary,
                'modified_by_doctor': doctor_made_edits
            })
        
        return jsonify({
            'success': False,
            'message': 'Invalid action'
        }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error processing approval: {str(e)}"
        }), 500
    
def send_approval_email(doctor_email, patient_id, summary, approval_link):
    """Send approval email to doctor"""
    try:
        if not EMAIL_USER or not EMAIL_PASSWORD:
            print("Email credentials not configured")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = doctor_email
        msg['Subject'] = f"Discharge Summary Approval Required - Patient {patient_id}"
        
        # Create email body
        body = f"""
        Dear Doctor,
        
        A discharge summary for Patient ID: {patient_id} requires your approval.
        
        Summary Preview:
        {summary[:300]}...
        
        Please click the link below to review and approve/modify the summary:
        {approval_link}
        
        This link will expire in 24 hours.
        
        Best regards,
        Medical Records System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, doctor_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"Email sending error: {str(e)}")  # Log the specific error
        return False

def update_patient_record(patient_id, approved_summary):
    """Update patient record with approved summary"""
    try:
        # Create approved summaries directory
        output_dir = "approved_summaries"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save approved summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_approved_summary_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(approved_summary)
        
        print(f"Approved summary saved: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error updating patient record: {str(e)}")
        return False
    

@app.route('/api/check-approval-status/<patient_id>')
def check_approval_status(patient_id):
    """Check if there's an approval status update for a patient"""
    try:
        # Check if patient has any approved summaries
        for token, data in approved_summaries.items():
            if data['patient_id'] == patient_id:
                return jsonify({
                    'success': True,
                    'status': 'approved',
                    'final_summary': data['final_summary'],
                    'modified_by_doctor': data.get('modified_by_doctor', False),
                    'approved_at': data['approved_at']
                })
        
        return jsonify({
            'success': True,
            'status': 'pending'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error checking status: {str(e)}"
        }), 500
    

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
