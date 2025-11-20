from setuptools import setup, find_packages

setup(
        name='spookyflows',
        version='0.2.1',
        packages=find_packages(),
        description='Tools and algorithms for fluid flow simulations',
        url='https://github.com/PatricioClark/spooky',
        author='Patricio Clark Di Leoni',
        author_email='pclarkdileoni@udesa.edu.ar',
        license='MIT',
        install_requires=['numpy'],
)
