from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="torchsvm",
    version="0.1.2",
    description="""torchsvm, a PyTorch-based library that trains kernel SVMs and other large-margin classifiers 
                   with exact leave-one-out cross-validation (LOOCV) error computation. Conventional SVM solvers 
                   often face scalability and efficiency challenges, especially on large datasets or when multiple 
                   cross-validation runs are required. 
                   torchsvm computes LOOCV at the same cost as training a single SVM, while further boosting speed 
                   and scalability via CUDA-accelerated matrix operations. """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yikai Zhang",
    packages=find_packages(include=["torchsvm"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    license="MIT",
    url="https://github.com/YikaiZhang95/torchsvm",
)
