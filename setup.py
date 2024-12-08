from setuptools import setup, find_packages

setup(
    name="Path_Development_Net",
    version="0.0.2",
    author="PDevNet",
    description="Path Development Network with Finite dimensional Lie Group",
    url="https://github.com/PDevNet/DevNet",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=['matplotlib',
                      'numpy',
                      'PyYAML',
                      'scikit_learn',
                      'scipy',
                      'torch',
                      'sktime',
                      'torchaudio',
                      'torchvision',
                      'seaborn',
                    #   'ml_collections',
                    #   'signatory', # ! requires torch installed first 
                      'tqdm'],
)
