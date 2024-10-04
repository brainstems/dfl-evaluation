import boto3
from botocore.exceptions import ClientError
import os
import logging
import pandas as pd
from io import BytesIO

def upload_text_to_s3(df, bucket, aws_access_key_id, aws_secret_access_key, aws_region, object_name, eval_method, columns):
    """
    Generate text content with DataFrame evaluation and upload to S3

    :param df: DataFrame to evaluate
    :param bucket: S3 bucket name
    :param aws_access_key_id: AWS access key ID
    :param aws_secret_access_key: AWS secret access key
    :param aws_region: AWS region name
    :param object_name: S3 object name for the uploaded file
    :param eval_method: Evaluation method description
    :param columns: List of columns to evaluate
    :return: True if the file was uploaded, else False
    """
    # Prepare the content to be written
    content = f"Evaluation method: {eval_method}\n"
    for col in columns:
        if col in df.columns:
            mean_value = df[col].mean()
            content += f"Mean of {col}: {mean_value}\n"
        else:
            content += f"Warning: Column '{col}' not found in DataFrame.\n"

    # Set up boto3 session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Upload the content to S3 using a buffer
    buffer = BytesIO(content.encode('utf-8'))
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_fileobj(buffer, bucket, object_name)
        print(f"Mean distances have been uploaded to {object_name} in bucket {bucket}")
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_dataframe_to_s3(
    dataframe,
    bucket,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
    object_name,
):
    """Upload a DataFrame as a Parquet file to an S3 bucket

    :param dataframe: DataFrame to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name
    :return: True if file was uploaded, else False
    """
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Save DataFrame to a buffer as Parquet
    buffer = BytesIO()
    dataframe.to_parquet(buffer, index=False)
    buffer.seek(0)  # Rewind the buffer

    # Upload the buffer to S3
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_fileobj(buffer, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(
    file_name,
    bucket,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
    object_name=None,
):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_JSON_to_dataframe(
    bucket,
    object_name,
    aws_access_key_id,
    aws_secret_access_key,
    aws_region,
):
    """Download an JSON file from S3 and stream it into a pandas DataFrame.

    :param bucket: Bucket to download from
    :param object_name: S3 object name to download
    :return: DataFrame containing the data from the object, or None if an error occurs
    """
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    s3_client = boto3.client("s3")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=object_name)

        # Load the file directly into a pandas DataFrame
        df = pd.read_json(BytesIO(obj["Body"].read()), lines=True)
        return df
    except Exception as e:
        return None


def download_PQT_file_to_dataframe(
    bucket, object_name, aws_access_key_id, aws_secret_access_key, aws_region
):
    """Download an Parquet file from S3 and stream it into a pandas DataFrame.

    :param bucket: Bucket to download from
    :param object_name: S3 object name to download
    :return: DataFrame containing the data from the object, or None if an error occurs
    """
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Download the Parquet file object
    s3_client = boto3.client("s3")
    try:
        # Download the Parquet file into memory
        obj = s3_client.get_object(Bucket=bucket, Key=object_name)

        # Load the Parquet file directly into a pandas DataFrame
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        return df
    except Exception as e:
        logging.error(e)
        return None

def download_CSV_to_dataframe(
    bucket, object_name, aws_access_key_id, aws_secret_access_key, aws_region
):
    """Download an CSV file from S3 and stream it into a pandas DataFrame.

    :param bucket: Bucket to download from
    :param object_name: S3 object name to download
    :return: DataFrame containing the data from the object, or None if an error occurs
    """
    # Set Session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # Download the Parquet file object
    s3_client = boto3.client("s3")
    try:
        # Download the Parquet file into memory
        obj = s3_client.get_object(Bucket=bucket, Key=object_name)

        # Load the Parquet file directly into a pandas DataFrame
        df = pd.read_csv(BytesIO(obj["Body"].read()))
        return df
    except Exception as e:
        logging.error(e)
        return None
