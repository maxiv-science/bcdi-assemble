import setuptools

setuptools.setup(
    name="bcdiass",
    version="0.0.1",
    author="Alexander Bjorling",
    author_email="alexander.bjorling@maxiv.lu.se",
    description="Diffraction volume assembly for Bragg CDI",
    url="https://github.com/maxiv-science/bcdi-assemble",
    packages=setuptools.find_packages(),
    install_requires=['h5py', 'ipython', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
