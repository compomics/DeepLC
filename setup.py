from setuptools import setup


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


setup(
    name='deeplc',
    version='0.1.20',
    license='apache-2.0',
    description='DeepLC: Retention time prediction for (modified) peptides using Deep Learning.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Robbin Bouwmeester, Niels Hulstaert, Ralf Gabriels, Prof. Lennart Martens, Prof. Sven Degroeve',
    author_email='Robbin.Bouwmeester@UGent.be',
    url='http://compomics.github.io/projects/DeepLC',
    project_urls={
        'Documentation': 'http://compomics.github.io/projects/DeepLC',
        'Source': 'https://github.com/compomics/DeepLC',
        'Tracker': 'https://github.com/compomics/DeepLC/issues'
    },
    packages=['deeplc'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['deeplc=deeplc.__main__:main']
    },
    keywords=[
        'DeepLC', 'Proteomics', 'deep learning', 'peptides', 'retention time',
        'prediction'
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 4 - Beta"
    ],
    install_requires=[
        'setuptools>=42.0.1',
        'tensorflow>=2.2,<3',
        'scipy>=1.4.1,<2',
        'numpy>=1.17,<2',
        'pandas>=0.25,<2',
        'matplotlib>=3,<4',
    ],
    python_requires='>=3.6',
)
