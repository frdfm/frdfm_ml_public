from setuptools import setup, find_packages

print("This message should appear when setup runs.")
print(find_packages())

setup(
    name='frdfmml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
)
