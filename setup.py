from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from a file, handling comments and empty lines."""
    requirements_path = Path(__file__).parent / filename
    if not requirements_path.exists():
        return []
    
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                # Remove inline comments
                line = line.split('#')[0].strip()
                if line:
                    requirements.append(line.split(' ')[0])
        return requirements

# Read different requirement files
install_requires = read_requirements('requirements.txt')

setup(
    name="fladrec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    # python_requires=">=3.11",
)