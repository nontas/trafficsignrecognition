from setuptools import setup, find_packages
import versioneer


setup(
    name='trafficsignrecognition',
    version='0.1',
    cmdclass=versioneer.get_cmdclass(),
    description="Traffic Sign Detection and Recognition",
    author='Epameinondas Antonakos',
    author_email='antonakosn@gmail.com',
    packages=find_packages(),
    install_requires=['menpo>=0.7,<0.8',
                      'scikit-learn>=0.17']
)
