import os
import re
import subprocess
from setuptools import find_packages, setup


def gitversion() -> str:
  _re = re.compile('^Version: (.+)$', re.M)
  d = os.path.dirname(__file__)
  if os.path.isdir(os.path.join(d, '.git')):
    # Get the version using "git describe".
    cmd = 'git describe --tags --match [0-9]*'.split()
    try:
      version = subprocess.check_output(cmd).decode().strip()
    except subprocess.CalledProcessError:
      print('Unable to get version number from git tags')
      exit(1)
    # PEP 386 compatibility
    if '-' in version:
      version = '.post'.join(version.split('-')[:2])
    # Don't declare a version "dirty" merely because a time stamp has changed. 
    # If it is dirty, append a ".dev1" suffix to indicate a development revision after the release.
    with open(os.devnull, 'w') as fd_devnull:
      subprocess.call(['git', 'status'], stdout=fd_devnull, stderr=fd_devnull)
    cmd = 'git diff-index --name-only HEAD'.split()
    try:
      dirty = subprocess.check_output(cmd).decode().strip()
    except subprocess.CalledProcessError:
      print('Unable to get git index status')
      exit(1)
    if dirty != '':
      version += '.dev1'
  else:
    # Extract the version from the PKG-INFO file.
    with open(os.path.join(d, 'PKG-INFO')) as f:
      version = _re.search(f.read()).group(1)
  return version


setup(
  name="trism",
  version=gitversion(),
  packages=find_packages(),
  license="GNU AGPL v3.0",
  zip_safe=True,
  description="Simple Python package that helps us infer with Triton Inference Server easily",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  author="Hieu Pham",
  author_email="64821726+hieupth@users.noreply.github.com",
  url="https://github.com/hieupth/pytrism",
  install_requires=['numpy', 'tritonclient'],
  classifiers=['Intended Audience :: Developers', 'Topic :: Software Development :: Build Tools']
)