from setuptools import setup


setup(
    name='nucleus',
    version='0.0.1',
    author='hudlrd',
    install_requires=[
        'stringcase==1.2.0',
        'tqdm==4.31.1',
        'boto3==1.9.114',

        'altair==2.4.1',
        'matplotlib==3.0.3',
        'pydot==1.4.1',
        'graphviz==0.10.1',

        'pillow==5.4.1',

        'quilt==2.9.15',

        'pandas==0.24.1',

        # Hudl packages
        'hudl-aws==0.2.1',
    ],
    extras_require={
        'cpu': ['tf-nightly-2.0-preview'],
        'gpu': ['tf-nightly-gpu-2.0-preview'],
    },
    test_require={
        'pytest',
        'pytest-cov',
        'coverage'
    },
)
