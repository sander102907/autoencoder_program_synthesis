from setuptools import setup, find_packages
import autoencoder_program_synthesis

setup(name='Autoencoder Program Synthesis',
      version=autoencoder_program_synthesis.__version__,
      description='Autoencoder Program Synthesis: A Tree-based VAE-RNN as a tool for generating new C++ programs.',
      author='Sander de Bruin',
      author_email='s.d.m.d.bruin@student.tue.nl',
      url='https://github.com/sander102907/autoencoder_program_synthesis',
      packages=find_packages(),
      install_requires=[
          'anytree',
          'cpp_ast_parser @ git+https://github.com/sander102907/cpp_ast_parser',
          'nltk',
          'sacred',
          'sctokenizer',
          'sacredboard',
          'gensim',
          'pytorch-tree-lstm',
          'numpy',
          'pymongo',
          'icecream',
      ],
      long_description=open('README.md').read(),
      license='APACHE 2.0'
     )
