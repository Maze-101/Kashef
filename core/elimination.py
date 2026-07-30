import numpy as np

def partial_pivot(A, b, current_row):    
    column_slice = A[current_row:, current_row];
    
    max_index_in_slice = np.argmax(np.abs(column_slice));
    
    max_row = current_row + max_index_in_slice;
    
    if max_row != current_row:
        A[[current_row, max_row]] = A[[max_row, current_row]];
        b[[current_row, max_row]] = b[[max_row, current_row]];
        
    return A, b

def forward_elimination(A, b):
    A = A.astype(np.float64);
    b = b.astype(np.float64);

    n = A.shape[0];

    for i in range(n):
        A, b = partial_pivot(A, b, i);

        if np.isclose(A[i,i], 0.0):
            raise ValueError("Matrix is singular.");
    
        for j in range(i + 1, n):
            factor = A[j,i] / A[i,i];
            A[j,i:] -= factor * A[i,i:];
            b[j] -= factor * b[i];

    return A,b;

def back_substitution(U, y):
    x = np.zeros_like(y);

    n = U.shape[0];

    for i in range(n - 1, -1, -1):
    
        x[i] = (y[i] - (U[i,i+1:] @ x[i+1:])) / U[i,i];
    
    return x;