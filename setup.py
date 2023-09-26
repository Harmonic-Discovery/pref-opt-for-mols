from setuptools import setup, find_packages

__version__ = "1.0"

setup(
    name="pref_opt_for_mols",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    description="GPT, CharRNN + preference optimization with DPO",
)

"""
TMPDIR=/home/ryan/pip_cache/ pip install --cache-dir=/home/ryan/pip_cache/ torch
"""
