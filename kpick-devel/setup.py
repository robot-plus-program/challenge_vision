import setuptools
from Cython.Build import cythonize
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

# def get_ext_paths(root_dir, exclude_files):
#     """get filepaths for compilation"""
#     paths = []
#
#     for root, dirs, files in os.walk(root_dir):
#         for filename in files:
#             if os.path.splitext(filename)[1] != '.py':
#                 continue
#
#             file_path = os.path.join(root, filename)
#             if file_path in exclude_files:
#                 continue
#
#             paths.append(file_path)
#     return paths
ConvertToBin_list = ['kpick/pick/grip/*.py', 'kpick/pick/suction/*.py', 'kpick/pick/grip_inner/*.py',
                     'kpick/pick/grip3/*.py',
                     'kpick/pick/suction/*.py', 'kpick/matching/*.py', 'kpick/matching/qatm/*.py']

setuptools.setup(
    name='kpick',
    version='1.0.0',
    author="Keti starteam",
    author_email="bmtrungvp@gmail.com",
    description="Keti AI-based Detectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keti-ai",
    packages=setuptools.find_packages(),
    # ext_modules=cythonize(ConvertToBin_list,
    #                       compiler_directives={'language_level': 3}
    #                       ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='MIT',
    keywords=['VISION', 'AI', 'Deep Learning'],
    install_requires=[
        'numpy==1.22.4',
        'opencv-python',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'pytorch_lightning==1.8.6',
        'pyrealsense2',
        'imutils',
        'gdown'
    ],
)
