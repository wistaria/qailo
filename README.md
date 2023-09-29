# Qailo - 回路: Simple Quantum Circuit Simulator

### Design Policy

* Use only numpy (and pytest for unit testing).
* Use pure functions only (always return the same output for each input, with no side effects).
* No new classes are introduced.
* All functions must be less than 30 lines.

### Installation

* prerequisites
  * python 3.7 or later

* installatio
  ```bash
  $ pip install .
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
