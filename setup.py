from setuptools import setup, find_packages

setup(
    name='plmDCA',
    version='0.1.0',
    author='Gabriele Farné, Lorenzo Rosset, Saverio Rossi, Francesco Zamponi, Martin Weigt',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Python implementation of plmDCA',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/spqb/plmDCA',
    packages=find_packages(include=['plmDCA', 'plmDCA.*']),
    include_package_data=True,
    python_requires='>=3.10',
    license_files=["LICENSE"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'plmDCA=plmDCA.cli:main',
        ],
    },
    install_requires=[
        'adabmDCA>=0.3.3',
    ],
)
