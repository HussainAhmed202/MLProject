from typing import List

from setuptools import find_packages, setup


def get_requirements(filepath: str) -> list[str]:
    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="ml project",
    version="0.1",
    description="semester project for heart disease prediction",
    author="Hussain ahmed",
    author_email="h2002ahmed@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
