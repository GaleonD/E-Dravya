import json
import boto3
import joblib
from botocore.exceptions import ClientError
import uuid
import os

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table_standards = dynamodb.Table('authentic_herb_standards')
table_history = dynamodb.Table('herb_scan_history')

# Define the names of your files in the S3 bucket
BUCKET_NAME = 'e-tongue-ml-models' # CHANGE IF NEEDED
MODEL_KEY = 'rft_herb_model.pkl'
LABEL_ENCODER_KEY = 'label_encoder.pkl'
IMPUTER_KEY = 'imputer.pkl'
FEATURE_COLUMNS_KEY = 'feature_columns.json'

# Download a file from S3 to Lambda's temporary storage
def download_file_from_s3(key, local_path):
    try:
        s3.download_file(BUCKET_NAME, key, local_path)
        print(f"Successfully downloaded {key}")
        return True
    except ClientError as e:
        print(f"Error downloading {key} from S3: {e}")
        return False

# Load the model and artifacts (happens once when Lambda container starts)
def load_artifacts_from_s3():
    artifacts = {}
    
    # Download files
    download_file_from_s3(MODEL_KEY, '/tmp/model.pkl')
    download_file_from_s3(LABEL_ENCODER_KEY, '/tmp/le.pkl')
    download_file_from_s3(IMPUTER_KEY, '/tmp/imputer.pkl')
    download_file_from_s3(FEATURE_COLUMNS_KEY, '/tmp/feature_columns.json')
    
    # Load the files into memory
    artifacts['model'] = joblib.load('/tmp/model.pkl')
    artifacts['label_encoder'] = joblib.load('/tmp/le.pkl')
    artifacts['imputer'] = joblib.load('/tmp/imputer.pkl')
    
    with open('/tmp/feature_columns.json', 'r') as f:
        artifacts['feature_columns'] = json.load(f)
        
    print("All model artifacts loaded successfully")
    return artifacts

# Load the artifacts into memory
artifacts = load_artifacts_from_s3()

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event))

    # 1. Parse incoming request
    try:
        body = json.loads(event['body'])
        sensor_readings = body['sensor_readings'] # e.g., {'pH': 6.5, 'TDS_ppm': 250, ...}
    except (KeyError, json.JSONDecodeError) as e:
        return {'statusCode': 400, 'body': json.dumps(f'Invalid request: {e}')}

    # 2. Prepare the feature vector in the EXACT order the model expects
    try:
        feature_vector = []
        for feature in artifacts['feature_columns']:
            # Get the value from the sensor readings, use 0 if not provided
            value = sensor_readings.get(feature, 0.0)
            feature_vector.append(float(value))
    except Exception as e:
        return {'statusCode': 400, 'body': json.dumps(f'Error preparing features: {e}')}

    # 3. Apply preprocessing (imputer)
    try:
        feature_vector = artifacts['imputer'].transform([feature_vector])
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(f'Error in imputer: {e}')}

    # 4. Make a prediction
    try:
        prediction_numeric = artifacts['model'].predict(feature_vector)
        herb_name = artifacts['label_encoder'].inverse_transform(prediction_numeric)[0]
        
        # Get prediction confidence (if available)
        if hasattr(artifacts['model'], 'predict_proba'):
            confidence = artifacts['model'].predict_proba(feature_vector).max()
        else:
            confidence = 1.0 # Placeholder if model doesn't support probabilities
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps(f'Model prediction failed: {e}')}

    # 5. Get Golden Standard for this herb
    try:
        response = table_standards.get_item(Key={'herb_name': herb_name})
        golden_data = response['Item']
    except ClientError as e:
        return {'statusCode': 500, 'body': json.dumps(f'Error fetching golden standard: {e}')}

    # 6. Check for Adulteration (Example: Check TDS)
    live_tds = sensor_readings.get('TDS_ppm', 0)
    golden_tds = golden_data.get('avg_tds', 1) # Avoid division by zero
    tds_ratio = live_tds / golden_tds
    quality_threshold = golden_data.get('quality_threshold', 0.7)
    is_potent = tds_ratio >= quality_threshold

    # 7. Save to History
    try:
        scan_id = str(uuid.uuid4())
        timestamp = int(time.time())
        
        table_history.put_item(
            Item={
                'scan_id': scan_id,
                'timestamp': timestamp,
                'sensor_data': json.dumps(sensor_readings),
                'prediction': herb_name,
                'confidence': confidence,
                'is_potent': is_potent,
                'adultaration_alert': not is_potent
            }
        )
    except ClientError as e:
        print(f"Warning: Could not save to history table: {e}")

    # 8. Return Response
    response_payload = {
        "predicted_herb": herb_name,
        "confidence": confidence,
        "adultaration_alert": not is_potent,
        "message": "Quality check passed." if is_potent else "WARNING: Potency below threshold."
    }

    return {
        'statusCode': 200,
        'headers': { 
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(response_payload)
    }