"""Command-line interface for solving ``Ax = b`` by Gaussian elimination."""

import numpy as np

from core.elimination import back_substitution, forward_elimination


def read_positive_integer(prompt):
    """Read a positive integer, prompting again after invalid input."""
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer.")


def read_row(prompt, width):
    """Read one numeric row with exactly ``width`` values."""
    while True:
        try:
            row = [float(value) for value in input(prompt).split()]
            if len(row) == width:
                return row
        except ValueError:
            pass
        print(f"Please enter exactly {width} numeric value(s), separated by spaces.")


def main():
    print("Gaussian elimination solver: Ax = b")
    n = read_positive_integer("Number of equations (and unknowns): ")

    print(f"Enter the {n} rows of coefficient matrix A ({n} values per row):")
    A = np.array(
        [read_row(f"A row {index + 1}: ", n) for index in range(n)], dtype=float
    )

    rhs_columns = read_positive_integer("Number of columns in b: ")
    print(f"Enter the {n} rows of b ({rhs_columns} value(s) per row):")
    b = np.array(
        [read_row(f"b row {index + 1}: ", rhs_columns) for index in range(n)],
        dtype=float,
    )

    # The elimination code accepts a one-dimensional vector for one RHS.
    rhs = b[:, 0] if rhs_columns == 1 else b

    try:
        upper_triangular, transformed_rhs = forward_elimination(A, rhs)
        solution = back_substitution(upper_triangular, transformed_rhs)
    except ValueError as error:
        print(f"Unable to solve the system: {error}")
        return

    print("Solution x:")
    print(solution)


if __name__ == "__main__":
    main()
