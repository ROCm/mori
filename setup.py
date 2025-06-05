from setuptools import find_packages, setup

setup(
    name="mori",
    version="0.0.0",
    description="Modular RDMA Interface",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    # package_data={},
    # cmdclass={},
    # options={"bdist_wheel": {"py_limited_api": "cp39"}},
    # zip_safe=False,
    install_requires=["torch"],
    python_requires=">=3.10",
    # ext_modules=extensions,
    # include_package_data=True,
)