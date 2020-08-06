from distutils.core import setup
setup(
  name = 'efsassembler',         
  packages = ['efsassembler'],  
  version = '0.0.1',
  license='GNU General Public License v3.0',
  description = 'A framework for building Ensemble Feature Selection experiments.',   
  author = 'Felipe Colombelli',                   
  author_email = 'fcolombelli@inf.ufrgs.br',
  url = 'https://github.com/colombelli/efs-assembler',
  keywords = ['ensemble', 'feature', 'selection', 'genes'],
  install_requires=[
          'pandas>=1.0.5',
          'numpy>=1.19.0',
          'rpy2>=3.3.4',
          'scikit-learn>=0.23.1',
          'ReliefF>=0.1.2',
          'tensorflow>=2.2.0',
          'keras>=2.4.3'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: GNU General Public License v3.0',
    'Programming Language :: Python :: 3.8',
  ],
)