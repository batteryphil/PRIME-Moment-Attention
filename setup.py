import os
from setuptools import setup, find_packages

# Optional C++ extension compilation
ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    cpp_source = os.path.join("src", "prime_moment_attention", "prime_cpp_kernel.cpp")
    if os.path.exists(cpp_source) and os.environ.get("PRIME_BUILD_CPP", "0") == "1":
        ext_modules.append(
            CppExtension(
                name="prime_moment_attention.prime_cpp_hybrid",
                sources=[cpp_source],
                extra_compile_args=["-O3"]
            )
        )
        cmdclass["build_ext"] = BuildExtension
except Exception:
    pass

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
