from setuptools import setup, find_packages

setup(
    name='hust-XGCN',
    version='0.0.0',
    description='Setting up a python package',
    author='Xiran Song',
    author_email='xiransong@hust.edu.cn',
    # url='https://blog.godatadriven.com/setup-py',
    packages=find_packages(include=['XGCN']),
    install_requires=[
        "dgl >= 0.9.1",
        "gensim >= 4.2.0",
        "numba >= 0.55.1",
        "numpy >= 1.21.5",
        "PyYAML >= 6.0",
        "torch >= 1.9.0",
        "torch_geometric >= 2.0.4",
        "tqdm >= 4.63.0",
    ],
    classifiers=["License :: OSI Approved :: MIT License"],
    # extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    # setup_requires=['pytest-runner', 'flake8'],
    # tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main']
    # },
    # package_data={'exampleproject': ['data/schema.json']}
)