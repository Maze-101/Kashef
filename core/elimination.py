import numpy as np

def partial_pivot(A, b, current_row):
    n = A.shape[0]
    
    # 1. Get all elements in the 'current_row' column, from 'current_row' down to the bottom
    column_slice = A[current_row:, current_row]
    
    # 2. Find the index of the largest ABSOLUTE value in that slice
    # np.argmax returns the index relative to the slice (starting at 0)
    max_index_in_slice = np.argmax(np.abs(column_slice))
    
    # 3. Convert that slice index back to the actual matrix row index
    max_row = current_row + max_index_in_slice
    
    # 4. Swap the rows if a larger pivot was found below
    if max_row != current_row:
        # NumPy trick to swap rows simultaneously in-place
        A[[current_row, max_row]] = A[[max_row, current_row]]
        b[[current_row, max_row]] = b[[max_row, current_row]]
        
    return A, b