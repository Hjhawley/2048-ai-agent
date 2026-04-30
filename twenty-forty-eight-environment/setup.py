from setuptools import setup, find_packages

setup(
    name='twenty_forty_eight_env',
    version='0.0.1',
    install_requires=['gymnasium', 'numpy', 'pygame'],
    packages=find_packages(),
    include_package_data=True,
)
