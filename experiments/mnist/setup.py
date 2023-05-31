from setuptools import setup

setup(
    name="official_mnist_tf",
    version="0.1.0",
    description="",
    author="Le, Tuan Anh <tuananh.le@dai-labor.de>",
    python_requires=">=3.6",
    install_requires=[
        # Add your dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=5.2",
        ]
    },
)
