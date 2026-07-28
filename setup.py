import os
from setuptools import setup, find_namespace_packages
from typing import List

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


ext_modules = []
CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

ops_extension = CUDAExtension(
    name="nanovllm.ops",
    sources=[
        "csrc/ops.cpp",
        "csrc/pos_encoding_kernels.cu",
        "csrc/layernorm_kernels.cu",
        "csrc/activation_kernels.cu",
        "csrc/moe_align_block_size_kernels.cu",
        "csrc/moe_topk_softmax_kernels.cu",
        "csrc/store_kvcache_kernels.cu",
        "csrc/custom_all_reduce_kernels.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    extra_link_args=['-Wl,--no-as-needed', '-lcuda'],
)
ext_modules.append(ops_extension)

setup(
    name="hover-inference",
    version="0.1.0",
    packages=find_namespace_packages(include=["nanovllm*"]),
    package_data={"nanovllm.layers.for_moe": ["configs/*.json"]},
    license_files=("LICENSE", "THIRD_PARTY_NOTICES.md"),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
)
