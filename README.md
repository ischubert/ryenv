# ryenv

This package contains example environments for reinforcement learning with python. The environments are simulations in PhysX based on rai and the corresponding python bindings rai-python.

## Setup

### Required manual setup
- `rai-python` has to be cloned into `$HOME/git/` and built there (use PhysX=1)

### Installation
Then, clone this repo and install the project with editable files:
```
$ pip install -e .
```
To just install (without cloning the package) you can also use:
```
$ pip install git+ssh://git@gitlab.com/ischubert/ryenv.git@master#egg=ryenv
```

## Example Environments
### disk_env

A simple environment of a disk being pushed on a 2D table by a 2D finger.