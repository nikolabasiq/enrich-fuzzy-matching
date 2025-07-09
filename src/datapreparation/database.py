import json
import os
from sqlalchemy import create_engine
import boto3
from botocore.exceptions import ClientError

# Retrieve the secret name from environment variables
SECRET_NAME = os.environ["SECRET_NAME"]

def get_connection():
    # Define the AWS region
    region_name = "ap-southeast-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        # Retrieve the secret value using the secret name
        get_secret_value_response = client.get_secret_value(
            SecretId=SECRET_NAME
        )
    except ClientError as e:
        # Handle exceptions when retrieving the secret value
        raise e

    # Extract the secret string from the response
    secret = get_secret_value_response['SecretString']

    # Parse the secret string as JSON
    secret_object = json.loads(secret)

    # Extract database credentials from the secret object
    database_username = secret_object['username']
    database_password = secret_object['password']
    database_hostname = secret_object['host']
    database_port = secret_object['port']
    database_name = secret_object['db_name']

    # Construct the database URL
    DATABASE_URL = (
        f"postgresql://{database_username}:{database_password}@{database_hostname}:{database_port}/{database_name}"
    )

    # Create and return a SQLAlchemy engine
    return create_engine(DATABASE_URL)