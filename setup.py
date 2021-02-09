# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")


setup(
    long_description=readme,
    name="utils",
    version="3.9.0",
    description="Alpha trading utils with pandas",
    python_requires="==3.*,>=3.7.0",
    project_urls={
        "homepage": "https://github.com/alpha-trading/utils",
        "repository": "https://github.com/alpha-trading/utils",
    },
    author="yangroro",
    author_email="yang@heechan.kr",
    license="MIT",
    packages=["utils"],
    package_dir={"": "."},
    package_data={},
    install_requires=[
        "pandas==1.*,>=1.1.0",
        "pymysql==0.*,>=0.9.0",
        "pypika==0.*,>=0.47.0",
        "python-dotenv==0.*,>=0.14.0",
        "python-telegram-bot==12.*,>=12.8.0",
    ],
    extras_require={"dev": ["dephell==0.*,>=0.8.3"]},
)
