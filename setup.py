from setuptools import setup, find_packages

setup(
        name='pySPEC',
        version='0.1',
        packages=find_packages(),
        description='Tools and algorithms from pseudo-spectral methods',
        url='https://github.com/PatricioClark/pySPEC',
        author='Patricio Clark Di Leoni',
        author_email='pclarkdileoni@udesa.edu.ar',
        license='MIT',
        install_requires=['numpy'],
)
