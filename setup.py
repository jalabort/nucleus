from setuptools import setup


setup(
    name='nucleus',
    version='0.0.1',
    author='hudlrd',
    install_requires=[
        'stringcase>=1.2.0',
        'tqdm>=4.32.2',
        'boto3>=1.9.114',

        'altair>=3.1.0',
        'matplotlib>=3.0.3',
        'pydot>=1.4.1',
        'graphviz>=0.10.1',
        
        'pillow>=5.4.1',

        'quilt==2.9.15',
        'quilt3==3.1.0',

        'pandas>=0.24.2',
        'tensorflow-addons>=0.5.0',

        # Hudl packages
        'hudl-aws>=0.2.5'
    ],
    extras_require={
        'cpu': ['tensorflow==2.0.0-rc0'],
        'gpu': ['tensorflow-gpu==2.0.0-rc0']
    },
    test_require={
        'pytest',
        'pytest-cov',
        'coverage'
    },
)
