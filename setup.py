from distutils.core import setup
setup(
  name = 'efsassemb',         
  packages = ['efsassemb'],  
  version = '0.0.1',
  license='GNU General Public License v3.0',
  description = 'A framework for building Ensemble Feature Selection experiments.',   
  author = 'Felipe Colombelli',                   
  author_email = 'fcolombelli@inf.ufrgs.br',
  url = 'https://github.com/colombelli/efs-assembler',
  keywords = ['ensemble', 'feature', 'selection', 'genes'],
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'rpy2',
          'scikit-learn',
          'ReliefF',
          'tensorflow',
          'keras'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: GNU General Public License v3.0',
    'Programming Language :: Python :: 3.8',
  ],
)