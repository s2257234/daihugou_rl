from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="agents._mcts_fast",
    sources=["agents/_mcts_fast.pyx"],
)

setup(
    name="daihugou_rl_mcts_fast",
    ext_modules=cythonize(
        [ext],
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
)