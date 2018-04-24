from distutils.core import setup

setup(
    name='rainbow',
    version='testing',
    url='https://github.com/ichaelm/ShrubPig/tree/master/agents/rainbow/',
    modules=[
        'dqn_algo',
        'rainbow_dqn_model',
        'sonic_util',
    ],
    install_requires=[
        'docker',
        'flask',
        'flask_api',
    ],
)
