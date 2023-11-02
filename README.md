[![Actions status](https://github.com/wistaria/qailo/actions/workflows/pytest.yml/badge.svg)]
(https://github.com/wistaria/qailo/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Qailo - 回路: Simple Quantum Circuit Simulator

### Design Policy

* Depends only on numpy and matplotlib (and pytest for unit testing).
* Pure functions only (always return the same output for each input, with no side effects).
* No new classes are introduced. (state vectors, density matrices, and operators are all numpy.array.)
* All functions are less than 30 lines.

### Installation

* prerequisites
  * python 3.8 or later

* installation
  ```bash
  $ pip install .
  ```

* for development
  ```bash
  $ pip install matplotlib numpy pytest black ruff wheel
  $ pip install --no-build-isolation -ve .
  ```

* tests
  ```bash
  $ pip install .[dev]
  $ pytest
  ```

* examples
  ```bash
  $ python3 example/shimada-2.2.py
  $ python3 example/grover.py
  ```

### Shape of arrays

* state vector
  - length: n + 1
  - shape: [2,2,...,2,1]

* density matrix
  - length: 2 * n + 2
  - shape: [2,2,...,2,1,1]

* operator
  - length: 2 * n
  - shape: [2,2,...,2]

where n is the number of qubits.
