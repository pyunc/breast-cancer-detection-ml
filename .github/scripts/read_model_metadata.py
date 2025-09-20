#!/usr/bin/env python3
import os
import sys
import boto3
import json

def main():
    print(f"[read_model_metadata] Starting metadata read process")
    
    # Get environment variables
    ak = os.environ.get('DO_SPACES_ACCESS_KEY')
    sk = os.environ.get('DO_SPACES_SECRET_KEY')
    bucket = os.environ.get('DO_SPACES_BUCKET')
    region = os.environ.get('DO_SPACES_REGION', 'fra1')
    github_output = os.environ.get('GITHUB_OUTPUT')
    
    print(f"[read_model_metadata] DO_SPACES_BUCKET: {bucket}")
    print(f"[read_model_metadata] DO_SPACES_REGION: {region}")
    print(f"[read_model_metadata] GITHUB_OUTPUT: {github_output}")
    print(f"[read_model_metadata] ACCESS_KEY exists: {bool(ak)}")
    print(f"[read_model_metadata] SECRET_KEY exists: {bool(sk)}")

    if not ak or not sk or not bucket:
        print('Missing required environment variables', file=sys.stderr)
        return 1
    
    if not github_output:
        print('Missing GITHUB_OUTPUT environment variable', file=sys.stderr)
        return 1

    # Connect to DigitalOcean Spaces
    endpoint_url = f'https://{region}.digitaloceanspaces.com'
    print(f"[read_model_metadata] Connecting to {endpoint_url}")
    
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=ak,
            aws_secret_access_key=sk
        )
        
        # Read metadata from bucket
        metadata_key = 'input/metadata/model_metadata.txt'
        print(f"[read_model_metadata] Reading from {bucket}/{metadata_key}")
        
        response = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata_content = response['Body'].read().decode('utf-8')
        
        print(f"[read_model_metadata] Raw content: {metadata_content}")
        
        # Parse JSON
        metadata = json.loads(metadata_content)
        
        model_url = metadata['model_url']
        preprocessor_url = metadata['preprocessor_url']
        
        print(f"[read_model_metadata] Parsed model_url: {model_url}")
        print(f"[read_model_metadata] Parsed preprocessor_url: {preprocessor_url}")
        
        # Write to GITHUB_OUTPUT
        with open(github_output, 'a', encoding='utf-8') as fh:
            fh.write(f'model_url={model_url}\n')
            fh.write(f'preprocessor_url={preprocessor_url}\n')
            fh.flush()
            os.fsync(fh.fileno())
        
        print(f"[read_model_metadata] Successfully wrote outputs to {github_output}")
        
        # Verify the write
        with open(github_output, 'r', encoding='utf-8') as fh:
            content = fh.read()
            if f'model_url={model_url}' in content and f'preprocessor_url={preprocessor_url}' in content:
                print(f"[read_model_metadata] Verified outputs were written correctly")
            else:
                print(f"[read_model_metadata] WARNING: Outputs not found in file after write")
        
        return 0
        
    except Exception as e:
        print(f'Error reading metadata: {e}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
