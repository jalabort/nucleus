from typing import Optional, Tuple, List

import re
import boto3


s3_path_regex = re.compile('s3://([^/]*)/(.*)')


def parse_s3_url(url: str) -> List[Optional[str]]:
    r"""
    Parse S3 Url into parts.

    Parameters
    ----------
    url
        Url to be broken up

    Returns
    -------
    bucket_name
    keys
    region
    """
    bucket = None
    region = None
    key = None

    # http://bucket.s3.amazonaws.com/key1/key2
    match = re.search('^https?://([^.]+).s3.amazonaws.com(.*?)$', url)
    if match:
        # No region specified means us-east-1
        bucket, region, key = match.group(1), 'us-east-1', match.group(2)

    # http://bucket.s3-aws-region.amazonaws.com/key1/key2
    match = re.search('^https?://([^.]+).s3-([^.]+).amazonaws.com(.*?)$', url)
    if match:
        bucket, region, key = (
            match.group(1), match.group(2), match.group(3))

    # http://s3.amazonaws.com/bucket/key1/key2
    match = re.search('^https?://s3.amazonaws.com/([^/]+)(.*?)$', url)
    if match:
        # No region specified means us-east-1
        bucket, region, key = match.group(1), 'us-east-1', match.group(2)

    # http://s3-aws-region.amazonaws.com/bucket/key1/key2
    match = re.search('^https?://s3-([^.]+).amazonaws.com/([^/]+)(.*?)$',
                      url)
    if match:
        bucket, region, key = (
            match.group(2), match.group(1), match.group(3))

    # http://s3.aws-region.amazonaws.com/bucket/key1/key2
    match = re.search('^https?://s3.([^.]+).amazonaws.com/([^/]+)(.*?)$', url)
    if match:
        bucket, region, key = (
            match.group(2), match.group(1), match.group(3))

    pieces = [bucket, region, key]
    if any(map(lambda p: p is None, pieces)):
        raise Exception("Error parsing S3 URL: "
                        "bucket: {}, region: {}, key: {}".format(
                            bucket, region, key))

    return list(map(lambda x: x.strip('/'), pieces))


def is_s3_path(path: str) -> bool:
    r"""

    Parameters
    ----------
    path

    Returns
    -------

    """
    return True if path.startswith('s3://') else False


def parse_s3_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses an s3 path of the form s3://<bucket>/<key> into bucket and
    key. Returns (None, None) if input path is not in the correct form.

    Parameters
    ----------
    path
        The s3 path to parse

    Returns
    -------
    bucket, key
        The parsed bucket and key.
    """
    match = s3_path_regex.match(path)
    if match is None:
        return None, None

    groups = match.groups()
    return groups[0], groups[1]


def get_signed_s3_url(
        s3_path: str,
        expiry_seconds: int = 3600
) -> Optional[str]:
    r"""
    Generates a signed S3 url for the S3File.
    This is necessary for exposing S3 file URLs publicly.

    Parameters
    ----------
    s3_path
        The path on S3 for the S3File.
    expiry_seconds
        Number of seconds until the signed URL should expire.

    Returns
    -------
    signed_url
        The signed S3 url.
    """

    s3_bucket, s3_key = parse_s3_path(s3_path)
    return boto3.client('s3', 'us-east-1').generate_presigned_url(
        'get_object',
        Params={'Bucket': s3_bucket, 'Key': s3_key},
        ExpiresIn=expiry_seconds
    ).replace('https://', 'http://', 1)
