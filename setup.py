"""
安装程序
"""
import ast
import codecs
import os
import re
import sys

from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')
_source_dir = os.path.split(os.path.abspath(__file__))[0]

# Python版本
if sys.version_info[0] == 2:
    def read_file(path):
        with open(path, 'rb') as f:
            return f.read()
else:
    def read_file(path):
        with codecs.open(path, 'rb', 'utf-8') as f:
            return f.read()

# 模块版本
version = str(ast.literal_eval(_version_re.search(read_file(os.path.join(_source_dir, 'donut/__init__.py'))).group(1)))

# 处理需求文件
requirements_list = list(filter(
    lambda v: v and not v.startswith('#'),
    (s.strip() for s in read_file(os.path.join(_source_dir, 'requirements.txt')).split('\n'))
))
suffix = "git+"
dependency_links = [s for s in requirements_list if s.startswith(suffix)]
install_requires = [s for s in requirements_list if not s.startswith(suffix)]

setup(
    name='donut-jill',
    version=version,
    url='https://github.com.cnpmjs.org/897615138/donut',
    license='MIT',
    author='JillW',
    author_email='897615138@qq.com',
    description='Donut',
    long_description=__doc__,
    packages=find_packages('.', include=['donut', 'donut.*']),
    zip_safe=False,
    platforms='any',
    setup_requires=['setuptools'],
    install_requires=install_requires,
    dependency_links=dependency_links,
    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
