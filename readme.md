# Samurai Euler Tutorial

This project demonstrates how to solve the Euler equations using the [Samurai](https://github.com/hpc-maths/samurai) library with adaptive mesh refinement (AMR).

## Prerequisites

You need a package manager like `conda` or `mamba` to install the dependencies.

## Installation

1.  **Create the Conda environment:**

    Use the provided `environment.yml` file to create the environment.

    ```bash
    mamba env create -f conda/environment.yml
    ```

    Or with conda:

    ```bash
    conda env create -f conda/environment.yml
    ```

2.  **Activate the environment:**

    ```bash
    mamba activate samurai-euler-env
    ```

## Building the Project

1.  **Create a build directory:**

    ```bash
    mkdir build
    cd build
    ```

2.  **Configure the project with CMake:**

    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Build the executables:**

    ```bash
    make
    ```

## Running the Tests

### Euler 2D

To run the 2D Euler simulation:

```bash
./euler_2d
```

This will generate output files (e.g., HDF5/XDMF) in the `results` directory (or current directory depending on configuration), which can be visualized using ParaView.

### Other Executables

*   `euler_1d`: 1D Euler simulation.
*   `euler_user_pred_1d`: 1D Euler simulation with user-defined prediction.

## Command Line Options

The `euler_2d` executable accepts several command-line arguments to control the simulation, multiresolution parameters, and output.

### Simulation Parameters

| Option           | Description                                     | Default                  |
| :--------------- | :---------------------------------------------- | :----------------------- |
| `--cfl`          | The CFL number                                  | `0.9`                    |
| `--Ti`           | Initial time                                    | `0.0`                    |
| `--Tf`           | Final time                                      | `0.25`                   |
| `--scheme`       | Finite volume scheme (`rusanov`, `hll`, `hllc`) | `hllc`                   |
| `--test-case`    | Test case to run                                | `double_mach_reflection` |
| `--restart-file` | Path to a file to restart the simulation from   | (empty)                  |

### Multiresolution Parameters

| Option        | Description                          | Default |
| :------------ | :----------------------------------- | :------ |
| `--min-level` | Minimum level of the multiresolution | `8`     |
| `--max-level` | Maximum level of the multiresolution | `8`     |

### Output Parameters

| Option     | Description                        | Default           |
| :--------- | :--------------------------------- | :---------------- |
| `--path`   | Output directory path              | Current directory |
| `--nfiles` | Number of output files to generate | `1`               |

### Example Usage

Run the simulation with a specific final time and output 10 files:

```bash
./euler_2d --Tf 0.5 --nfiles 10
```

Run with adaptive mesh refinement (levels 5 to 10):

```bash
./euler_2d --min-level 5 --max-level 10
```
