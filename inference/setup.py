from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cleaner",
        ["cpp/cleaner.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11']
    ),
]

setup(
    name="cleaner",
    ext_modules=ext_modules,
)
