#!/usr/bin/env python3
import os, sys, boto3
ENDPOINT = 'https://fra1.digitaloceanspaces.com'
BUCKET = 'breast-cancer-detection-ml'
PREFIX = 'models/'

# from dotenv import load_dotenv
# load_dotenv()

def main():
    print(f"[discover_latest_model] Starting discovery process")
    ak = os.environ.get('DO_SPACES_ACCESS_KEY')
    sk = os.environ.get('DO_SPACES_SECRET_KEY')
    Bucket = os.environ.get('DO_SPACES_BUCKET')
    github_output = os.environ.get('GITHUB_OUTPUT')

    print(f"[discover_latest_model] DO_SPACES_BUCKET: {Bucket}")
    print(f"[discover_latest_model] GITHUB_OUTPUT: {github_output}")
    print(f"[discover_latest_model] ACCESS_KEY exists: {bool(ak)}")
    print(f"[discover_latest_model] SECRET_KEY exists: {bool(sk)}")

    if not ak or not sk:
        print('Missing DO_SPACES_ACCESS_KEY/DO_SPACES_SECRET_KEY secrets', file=sys.stderr)
        return 1
    
    if not github_output:
        print('Missing GITHUB_OUTPUT environment variable', file=sys.stderr)
        return 1

    print(f"[discover_latest_model] Connecting to {ENDPOINT}")
    s3 = boto3.client('s3', endpoint_url=ENDPOINT, aws_access_key_id=ak, aws_secret_access_key=sk)
    newest = None
    cont = None
    obj_count = 0
    
    print(f"[discover_latest_model] Searching for models in {Bucket}/{PREFIX}")
    while True:
        kwargs = dict(Bucket=Bucket, Prefix=PREFIX, MaxKeys=1000)
        if cont:
            kwargs['ContinuationToken'] = cont
        try:
            resp = s3.list_objects_v2(**kwargs)
        except Exception as e:
            print(f'Error listing objects: {e}', file=sys.stderr)
            return 1
            
        contents = resp.get('Contents', []) or []
        obj_count += len(contents)
        print(f"[discover_latest_model] Found {len(contents)} objects in this batch, total so far: {obj_count}")
        
        for obj in contents:
            k = obj['Key']
            print(f"[discover_latest_model] Checking object: {k}")
            if k.endswith('logistic_regression.joblib'):
                print(f"[discover_latest_model] Found model: {k}, LastModified: {obj['LastModified']}")
                if newest is None or obj['LastModified'] > newest['LastModified']:
                    newest = obj
                    print(f"[discover_latest_model] New newest model: {k}")
                    
        if not resp.get('IsTruncated'):
            break
        cont = resp.get('NextContinuationToken')
        
    print(f"[discover_latest_model] Total objects scanned: {obj_count}")
    if not newest:
        print('No logistic_regression.joblib found', file=sys.stderr)
        return 1
        
    print(f"[discover_latest_model] Selected newest model: {newest['Key']}")
    folder = '/'.join(newest['Key'].split('/')[:-1])
    base = f'https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/{folder}'
    model = f'{base}/logistic_regression.joblib'
    prep = f'{base}/preprocessor.joblib'
    
    print('Discovered model_url:', model)
    print('Discovered preprocessor_url:', prep)
    
    # Write to GITHUB_OUTPUT with explicit flushing
    github_output = os.environ.get('GITHUB_OUTPUT')
    try:
        with open(github_output, 'a', encoding='utf-8') as fh:
            fh.write(f'model_url={model}\n')
            fh.write(f'preprocessor_url={prep}\n')
            fh.flush()  # Ensure data is written
            os.fsync(fh.fileno())  # Force OS to write to disk
        print(f"[discover_latest_model] Successfully wrote outputs to {github_output}")
        
        # Verify the write was successful
        with open(github_output, 'r', encoding='utf-8') as fh:
            content = fh.read()
            if f'model_url={model}' in content and f'preprocessor_url={prep}' in content:
                print(f"[discover_latest_model] Verified outputs were written correctly")
            else:
                print(f"[discover_latest_model] WARNING: Outputs not found in file after write")
                print(f"[discover_latest_model] File content: {content}")
                
    except Exception as e:
        print(f'Error writing to GITHUB_OUTPUT: {e}', file=sys.stderr)
        return 1
        
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
