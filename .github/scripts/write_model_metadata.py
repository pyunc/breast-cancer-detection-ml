#!/usr/bin/env python3
import os
import sys
import boto3
import json
from datetime import datetime

def main():
    print(f"[write_model_metadata] Starting metadata write process")
    
    # Get environment variables
    ak = os.environ.get('DO_SPACES_ACCESS_KEY')
    sk = os.environ.get('DO_SPACES_SECRET_KEY')
    bucket = os.environ.get('DO_SPACES_BUCKET')
    region = os.environ.get('DO_SPACES_REGION', 'fra1')
    
    # Get model folder from command line argument
    if len(sys.argv) != 2:
        print('Usage: write_model_metadata.py <model_folder>', file=sys.stderr)
        return 1
    
    model_folder = sys.argv[1]
    
    print(f"[write_model_metadata] Model folder: {model_folder}")
    print(f"[write_model_metadata] DO_SPACES_BUCKET: {bucket}")
    print(f"[write_model_metadata] DO_SPACES_REGION: {region}")
    print(f"[write_model_metadata] ACCESS_KEY exists: {bool(ak)}")
    print(f"[write_model_metadata] SECRET_KEY exists: {bool(sk)}")

    if not ak or not sk or not bucket:
        print('Missing required environment variables', file=sys.stderr)
        return 1

    # Construct URLs
    base_url = f'https://{bucket}.{region}.cdn.digitaloceanspaces.com/{model_folder}'
    model_url = f'{base_url}/logistic_regression.joblib'
    preprocessor_url = f'{base_url}/preprocessor.joblib'
    
    # Create metadata dictionary
    metadata = {
        "model_url": model_url,
        "preprocessor_url": preprocessor_url,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_folder": model_folder
    }
    
    print(f"[write_model_metadata] Metadata: {json.dumps(metadata, indent=2)}")
    
    # Write to DigitalOcean Spaces
    endpoint_url = f'https://{region}.digitaloceanspaces.com'
    print(f"[write_model_metadata] Connecting to {endpoint_url}")
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=ak,
            aws_secret_access_key=sk
        )
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata, indent=2)
        
        # Upload to bucket
        metadata_key = 'input/metadata/model_metadata.txt'
        print(f"[write_model_metadata] Uploading to {bucket}/{metadata_key}")
        
        s3.put_object(
            Bucket=bucket,
            Key=metadata_key,
            Body=metadata_json.encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"[write_model_metadata] Successfully uploaded metadata to {bucket}/{metadata_key}")
        
        # Verify the upload
        response = s3.head_object(Bucket=bucket, Key=metadata_key)
        print(f"[write_model_metadata] Verification: Object exists, size: {response['ContentLength']} bytes")
        
        return 0
        
    except Exception as e:
        print(f'Error uploading metadata: {e}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
