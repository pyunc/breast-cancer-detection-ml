#!/usr/bin/env python3
import os, sys, boto3
ENDPOINT = 'https://fra1.digitaloceanspaces.com'
BUCKET = 'breast-cancer-detection-ml'
PREFIX = 'models/'

# from dotenv import load_dotenv
# load_dotenv()

def main():
    ak = os.environ.get('DO_SPACES_ACCESS_KEY')
    sk = os.environ.get('DO_SPACES_SECRET_KEY')
    Bucket = os.environ.get('DO_SPACES_BUCKET')


    if not ak or not sk:
        print('Missing DO_SPACES_ACCESS_KEY/DO_SPACES_SECRET_KEY secrets', file=sys.stderr)
        return 1
    s3 = boto3.client('s3', endpoint_url=ENDPOINT, aws_access_key_id=ak, aws_secret_access_key=sk)
    newest = None
    cont = None
    while True:
        kwargs = dict(Bucket=Bucket, Prefix=PREFIX, MaxKeys=1000)
        if cont:
            kwargs['ContinuationToken'] = cont
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []) or []:
            k = obj['Key']
            if k.endswith('logistic_regression.joblib'):
                if newest is None or obj['LastModified'] > newest['LastModified']:
                    newest = obj
        if not resp.get('IsTruncated'):
            break
        cont = resp.get('NextContinuationToken')
    if not newest:
        print('No logistic_regression.joblib found', file=sys.stderr)
        return 1
    folder = '/'.join(newest['Key'].split('/')[:-1])
    base = f'https://breast-cancer-detection-ml.fra1.cdn.digitaloceanspaces.com/{folder}'
    model = f'{base}/logistic_regression.joblib'
    prep = f'{base}/preprocessor.joblib'
    print('Discovered model_url:', model)
    print('Discovered preprocessor_url:', prep)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        fh.write(f'model_url={model}\n')
        fh.write(f'preprocessor_url={prep}\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
