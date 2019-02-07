from setuptools import setup

setup(name='labtools',
      version=0.1,
      description="Tools to run experiments of the choMORE setup.",
      # https://pypi.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Manufacturing',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Natural Language :: English',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
          'Topic :: Multimedia :: Video :: Capture',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      author='V.T. Tenner',
      author_email='v.t.tenner@vu.nl',
      url='',
      license='GPL-3.0',
      packages=[
          'labtools',
      ],
      zip_safe=False,
      install_requires=[
          'labtools',
          'pyqtgraph',
          'astropy',
          'filename_tools',
          'tests',
          'numpy',
      ],
      extras_requires={
          'dev': [
              'pytest',
          ]
      }
      )

# python3 -m pip install --user --upgrade setuptools wheel twine
# python3 setup.py sdist bdist_wheel
# python3 -m twine upload dist/*
