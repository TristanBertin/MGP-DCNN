from setuptools import setup

setup(name='MGPDCNN',
      version='1.0',
      description='Implementation of MGP-DCNN',
      url='https://github.com/TristanBertin/MGPDCNN',
      author='Tristan BERTIN',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['MGPDCNN'],
      install_requires=[
                        'h5py',
                        'keras',
                        'numpy>=1.16',
                        'matplotlib',
                        'scikit-learn',
                        'gpytorch',
                        'scikit-optimize',
                        'tensorflow'
                        ],
      zip_safe=False,
      # test_suite='nose.collector',
      # tests_require=['nose'],
      )



