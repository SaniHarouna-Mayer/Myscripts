import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myscripts-st",
    version="0.1",
    author="Songsheng Tao",
    author_email="songshengtao1@gmail.com",
    description="My frequently used scripts in PDF analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/st3107/Myscripts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)