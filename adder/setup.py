from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

setup(
    name='adder',
    ext_modules=[
        CUDAExtension('adder_cuda', [
            'adder_cuda.cpp',
            'adder_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
