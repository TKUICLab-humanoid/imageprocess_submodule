from setuptools import find_packages, setup

package_name = 'imageprocess'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iclab',
    maintainer_email='612470079@o365.tku.edu.tw',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image = imageprocess.image:main',
            'image_test = imageprocess.image_test:main',
            'imageprocess = imageprocess.imageprocess:main',
        ],
    },
)
