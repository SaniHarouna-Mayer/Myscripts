# Myscripts

Welcome to Myscripts. The toolbox containing all the tools I used in the PDF analysis.

## Installation

Fork and clone this repo. Inside the local repo, run the pip install in the development mode.

``
pip install -e .
``

The development mode is suggested because the code is updated throughout my PhD career and my projects used different
version of the code. For using the older versions of the code, use git to checkout a new branch.

First, fetch all tags.
``
git fetch --all --tags
``
Then, checkout a new branch of the older version according to the tag.
``
git checkout tags/<tag name> -b <branch name>
``

## Examples

Please read the example scripts in the [examples](examples) folder to learn about how to use the tools in myscripts.
