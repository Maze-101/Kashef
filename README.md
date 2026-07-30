# Kashef - Linear Systems Solver with NumPy

A lightweight Python implementation of Gaussian Elimination with partial pivoting and back-substitution. This project builds a linear system solver from scratch using only core NumPy matrix slicing and vectorized operations, avoiding high-level abstractions like `np.linalg.solve`.

---
## Requirements

- Python 3.x
- NumPy

---
## Project Structure

The project follows a clean engineering layout separating core algorithm stages, helper functions, and runtime execution scripts:

```text
kashef/
│
├── core/
│   ├── elimination.py      # Core math logic (pivoting, elimination, back-substitution)
├── main.py                 # Interactive execution & demo test scripts
└── README.md               # Documentation and mathematical background
```
---
## Mathematical Background

A system of $n$ linear equations with $n$ variables is represented in matrix notation as:

$$Ax = b$$

Where:

- $A$ is an $n \times n$ coefficient matrix.
    
- $x$ is the $n \times 1$ unknown solution vector.
    
- $b$ is the $n \times 1$ constant output vector.
    

The algorithm progresses through three sequential mathematical phases to solve the system.

### 1. Partial Pivoting (Row Interchanging)

To ensure numerical stability and prevent division by zero, the solver evaluates the current working column before eliminating coefficients. At step $i$, it searches for the row $m$ (where $m \ge i$) containing the maximum absolute magnitude:

$$m = \arg\max_{i \le m < n} |A_{mi}|$$

If $m \neq i$, an elementary row interchange operation ($R_i \leftrightarrow R_m$) is applied simultaneously to both matrix $A$ and vector $b$:

$$A_{i, \dots} \leftrightarrow A_{m, \dots} \quad \text{and} \quad b_i \leftrightarrow b_m$$

### 2. Forward Elimination

The objective of forward elimination is to apply row modifications that reduce matrix $A$ to an Upper Triangular Matrix ($U$). For each row $j$ below the pivot row $i$ (where $j > i$), a scalar multiplier $c$ is computed:

$$c = \frac{A_{ji}}{A_{ii}}$$

The row transformation is then applied across the entire row slice to force all entries below the diagonal to zero:

$$R_j \leftarrow R_j - c \cdot R_i$$

$$b_j \leftarrow b_j - c \cdot b_i$$

If at any point a diagonal pivot element $A_{ii} \approx 0$ even after partial pivoting, the matrix is mathematically **singular** ($\det(A) = 0$), meaning a unique solution does not exist.

### 3. Back-Substitution

Once the system is converted into the upper triangular form ($Ux = y$), the unknown variables are solved sequentially in reverse order from $i = n-1$ down to $0$. The formula isolates the target unknown variable by subtracting the dot product of already known solution variables from the output constant:

$$x_i = \frac{y_i - \sum_{k=i+1}^{n-1} U_{ik}x_k}{U_{ii}}$$

---
## Core Program Workflow

1. **`partial_pivot(A, b, current_row)`**: Identifies the column's largest absolute value below the working row using `np.argmax(np.abs(...))` and executes an in-place simultaneous row swap.
    
2. **`forward_elimination(A, b)`**: Casts the input arrays to floating point values, loops across each column to secure pivot stability, detects singularity exceptions via `np.isclose()`, and applies vectorized row subtractions.
    
3. **`back_substitution(U, y)`**: Steps backward through indices via `range(n - 1, -1, -1)` and uses the NumPy matrix multiplication operator (`@`) to compute vector dot products recursively.
    
