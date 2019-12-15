from distutils.core import setup


LONG_DESCRIPTION = """\
DeepLC is a retention time predictor for (modified) peptides that employs Deep
Learning. It’s strength lies in the fact that it can accurately predict
retention times for modified peptides, even if hasn’t seen said modification
during training.

DeepLC can be run with a graphical user interface (GUI) or as a Python package.
In the latter case, DeepLC can be used from the command line,
or as a python module.
"""

setup(
    name='deeplc',
    version='0.1.1-dev5',
    license='apache-2.0',
    description='DeepLC: Retention time prediction for (modified) peptides using Deep Learning.',
    long_description=LONG_DESCRIPTION,
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
        'tensorflow>=1.14.0<3',
        'xgboost>=0.90,<2',
        'scipy>=1.3.1,<2',
        'matplotlib>=3,<4',
        'numpy>=1.17,<2',
        'pandas>=0.25,<1',
    ],
    python_requires='>=3.6,<3.8',
)
