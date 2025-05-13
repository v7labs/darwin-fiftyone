from setuptools import setup, find_packages

# Read requirements while filtering out comments and development dependencies
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not any(
            dev_dep in line
            for dev_dep in ["pytest", "black", "flake8", "coverage", "pytest-cov"]
        )
    ]

setup(
    name="darwin-fiftyone",
    version="1.1.23",
    description="Integration between V7 Darwin and Voxel51",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Simon Edwardsson & Mark Cox-Smith",
    packages=find_packages(),
    url="https://github.com/v7labs/darwin_fiftyone",
    install_requires=requirements,
)
