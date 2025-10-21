
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.3.1"
__version__ = version
full_version = version

git_revision = "4d833e5df760c382f24ee3eb643dc20c3da4a5a1"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
