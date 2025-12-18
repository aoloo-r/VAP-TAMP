from setuptools import setup

package_name = 'stretch_adapters'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/segway_bridge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aoloo',
    maintainer_email='aoloo@binghamton.edu',
    description='Adapters for using Stretch AI with Segway robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segway_observation_adapter = stretch_adapters.segway_observation_adapter:main',
        ],
    },
)