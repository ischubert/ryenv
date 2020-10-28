# ryenv

This package contains a collection of environments for reinforcement learning with python. The environments are simulations in PhysX based on [rai](https://github.com/MarcToussaint/rai) and the corresponding python bindings [rai-python](https://github.com/MarcToussaint/rai-python).

## Setup

### Required manual setup
- Install `rai-python` (configure PhysX=1 in config.mk)

### Installation
Clone this repo and install the project with editable files:
```
$ pip install -e .
```
To just install (without cloning the package) you can also use:
```
$ pip install git+ssh://git@github.com/ischubert/ryenv.git@master#egg=ryenv
```

## Environments
### DiskEnv

A simple environment of a disk being pushed on a 2D table by a 2D finger.

### DiskMazeEnv

Like `DiskEnv`, but there is a randomly generated maze on the table.

### BoxEnv

Box on a table, 10DoF

### PickAndPlaceEnv

Disk on a table, with "sticky" pick-and-place
