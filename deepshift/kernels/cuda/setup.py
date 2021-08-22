from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deepshift_cuda',
    ext_modules=[
        CUDAExtension('deepshift_cuda', [
            'shift_cuda.cpp',
            'shift.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


