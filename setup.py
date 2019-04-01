from setuptools import setup

setup(name='MGP_TNN',
      version='1.0',
      description='Implementation of MGP_TNN',
      url='http://github.com/storborg/funniest',
      author='Tristan BERTIN',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['MGP_TNN'],
      install_requires=[
                        'h5py',
                        'keras',
                        'numpy',
                        'matplotlib',
                        'scikit-learn',
                        ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )



