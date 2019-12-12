from distutils.core import setup

setup(
	name='deeplc',
	version='0.1.1',
	description='DeepLC: Retention time prediction for (modified) peptides using Deep Learning.',
	author='Robbin Bouwmeester, Niels Hulstaert, Ralf Gabriels, Prof. Lennart Martens, Prof. Sven Degroeve',
	author_email='Robbin.Bouwmeester@UGent.be',
	url='https://www.github.com/CompOmics/DeepLC',
	packages=['deeplc'],
	include_package_data=True,
	entry_points={
		'console_scripts': ['deeplc=deeplc.__main__:main']
	},
	classifiers=[
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: Apache Software License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Bio-Informatics",
		"Development Status :: 4 - Beta"
	],
	install_requires=[
        'matplotlib>=3.1.1,<4',
        'numpy>=1.17.2,<2',
        'pandas>=0.25.1,<1',
        'scipy>=1.3.1,<2',
        'tensorflow>=1.14.0<3',
        'xgboost>=0.90,<2',
	],
	python_requires='>=3.6',
)
