from setuptools import setup, find_packages

setup(
    name='ai_researcher',
    version='0.1.0',
    description='AI-powered research paper generation and review',
    author='AI Research Team',
    author_email='zhuminjun@westlake.edu.cn',
    url='https://github.com/zhu-minjun/Researcher',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.48.2',
        'vllm>=0.7.2',
        'bibtexparser',
        'datasets',
        'matplotlib',
        'scikit-learn ',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
