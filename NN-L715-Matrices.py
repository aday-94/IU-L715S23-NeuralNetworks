""" Neural Networks-L715 Spring 2023 Week 2 Practical Andrew Davis - Matrices (https://cl.indiana.edu/~ftyers/courses/2023/Spring/L-715/practicals/matrices.html)

The purpose of this exercise is to implement basic matrix operations in Python without the aid of numpy. 
Numpy has built-in versions of these operations that we will be using, but if we want to really program from scratch 
we should be able to program them ourselves. Fortunately they are fairly straightforward to program.

You can choose how you want to implement the functions, either with a class or with some function prefix. """

""" Question 1: Create a matrix of rows by cols where each element in the matrix is initialized to 0 or 1 """

#Option A (D's Method):

def matrix_zeros(a,b):
    return [[0]*b for i in range(a)]

print(matrix_zeros(4,2))

""" #Option B (My method ; 3,0 or 3,1 makes it all empty sets rather than 0 hmm): 

def matrix_zeros(rows, cols):
    
    matrix = []
    
    for i in range(rows):
       
        matrix.append([])
        
        for j in range(cols):
            
            matrix[i].append(0)
   
    return(matrix)

print(matrix_zeros(4, 2)) """

""" #Option C (F's Method ; 3,0 or 3,1 makes it all empty sets rather than 0 hmm):

def matrix_zeros(rows, cols):
    matrix = [
            [0 for i in range(cols)]
                for j in range(rows)]
    return matrix

print(matrix_zeros(3, 0)) """

""" #Option D (N's Solution -- Prevents the empty sets if enter in 3, 0):

def matrix_zeros(shape, matrix=0): 
    while(shape!=[]):
        matrix = [matrix for j in range(shape[-1])]
        shape= shape[:-1]
    return matrix

print(matrix_zeros(4,2)) """

""" Question 2: Matrix Copy -- Return a copy of the input matrix with equivalent rows and cols. 
Tip: First create a matrix of 0 and then copy. """ 

#Option A:

def matrix_copy(matrix):
    return [row[:] for row in matrix]

""" #Option B:

def copy_matrix(a):
    m = len(a)
    n = (a[0])
    k = matrix_zeros(m,n)

    for i in range(m):
        for j in range(n):
            k[i][j]=a[i][j]
    return k """

""" Question 3: Print Matrix -- Print out a matrix one row at a time in a nicely readable format. For example:

A = [[1, 4], [2, 1]]
print_matrix(A)

Would give something like:

[[1, 4],
 [2, 1]] """

#Option A:

def print_matrix(matrix):
    print('[', end='')
    for i, row in enumerate(matrix):
        if i != 0:
            print(' ', end='')
        print('[', end='')
        for j, col in enumerate(row):
            if j != 0:
                print(', ', end='')
            print(col, end='')
        print(']', end='')
        if i+1 == len(matrix):
            print(']\n', end='')
        else:
            print(',\n', end='')

""" #Option B:

def print_matrix(matrix):

    # First use if/else to check if its a one dimensional matrix or not
    if any(isinstance(elem, int) for elem in matrix):
        print(matrix)

    else:
        print('[', end='')

        # If not a one dimensional matrix, we iterate thru the ROWS of the matrix
        for i, row in enumerate(matrix):

            # Then if the row is NOT the first row, we add a space
            if i != 0:
                print(' ', end='')

            print('[', end='') # Here we add a left bracket for the start of the row

            # Now, we iterate thru the columns
            for j, col in enumerate(row):

                print(col, end='') # We print the element of the column

                # Then, we have an if/else that adds a comma and space if this is NOT the last element of the row
                # Otherwise, we add right closing bracket
                if j != (len(row) - 1):
                    print(', ', end='')
                else:
                    print(']', end = '')
            # If the row is NOT the last row, add a comma 
            if i != (len(matrix) - 1):
                print(',')

        print(']') """

""" Question 4: Matrix Addition and Subtraction -- These are essentially the same only the operation is different (e.g. + or -). 
You should check that the input matrices are of the same dimensions.

A = [[1, 4], [2, 1]]
B = [[6, 4], [4, 8]]

matrix_add(A, B)

Should give:

[[7, 8], [6, 9]]

You can check your result by comparing against numpy:

numpy.add(A,B)
array([[7, 8],
       [6, 9]]) """

#Option A:

a = [[1, 4], [2, 1]]
b = [[6, 4], [4, 8]]

def matrix_add(a, b):
    ret = []
    if len(a) != len(b):
        raise ValueError('Matrices with different dimensions cannot be added. Please run the program again with different matrices.')
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            raise ValueError('Matrices with different dimensions cannot be added. Please run the program again with different matrices.')
        ret.append([i+j for i,j in zip(ra, rb)])
    return ret

""" #Option B:

A = [[1, 4], [2, 1]]
B = [[6, 4], [4, 8]]
result = [[0,0], [0,0]] """

""" def matrix_add(A, B):

    # iterate through rows
    for i in range(len(A)):
    # iterate through columns
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
        
    #for r in result:
        #print(r)
    
    return result

print(matrix_add(A,B)) """

""" #Option C:

def matrix_shape(A):
    shape = []
    while isinstance(A, list):
        shape.append(len(a))
        A = A[0]
    return shape

def matrix_add(A, B):
    if (matrix_shape(A) != matrix_shape(B)):
       raise ValueError('Cannot add arrays with different shapes.') 
    C = []
#    C = matrix_zeros(len(A), len(A[0]))
#    for i in range(len(A)):
#        for j in range(len(A[0])):
#            C[i][j] = A[i][j] + B[i][j]
    return [C.append(row + col)] """

""" Question 3: Matrix Multiplication -- This is a bit more tricky. In this case, the columns in matrix A must match the rows 
in matrix B. Note that the order of elements matters too, so matrix_multiply(A, B) will not give the same result as 
matrix_multiply(B, A). 

A = [[1, 4, 3], 
     [2, 1, 5]]
B = [[6, 4], 
     [4, 8], 
     [3, 5]]

matrix_multiply(A, B)

Should return:

[[31, 51],
 [31, 41]]"""

#Option A:

a = [[1, 4, 3], 
     [2, 1, 5]]

b = [[6, 4], 
     [4, 8], 
     [3, 5]]

def column(matrix, n):
    return [row[n] for row in matrix]

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('cannot dot product vectors of different lengths')
    return sum(a*b for a,b in zip(v1, v2))

def matrix_multiply(a, b):
    ret = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            row.append(dot_product(column(b, j), a[i]))
        ret.append(row)
    return ret

""" #Option B:

a = [[1, 4, 3], 
     [2, 1, 5]]

b = [[6, 4], 
     [4, 8], 
     [3, 5]]

result = [[0, 0],
          [0, 0]]

def matrix_multiplication(A,B):

    #iterate thru the rows
    for i in range(len(A)):
        #iterate thru the columns
        for j in range(len(B[0])):

            #result[i][j] = A[i][j] * B[i][j]
            
            for k in range(len(B)):

                result[i][j] += A[i][k] * B[k][j]
    
    return(result)

print(matrix_multiplication(A,B)) """

""" #Option C:

def matrix_multiply(A, B):
    C = matrix_zeros(len(A), len(B[0]))
    
    for k in range(len(B[0])):
        
        for i in range(len(A)):
            mult = 0
            for j in range(len(A[0])):
                mult += A[i][j] * B[j][k]
            C[i][k] = mult
    
    return C """

""" Question 4: Matrix Transposition -- To transpose a matrix is to take the columns and swap them with the rows, 
for example, given a matrix:

[[6, 4],
 [4, 8],
 [3, 5]]

Produce:

[[6, 4, 3],
 [4, 8, 5]] """

#Option A:

def transpose(matrix):
    return [column(matrix, i) for i in range(len(matrix[0]))]

""" #Option B:

def transpose_matrix(matrix):
    num_rows, num_cols = get_shape(matrix)

    transpose = []
    for i in range(len(matrix[0])): # num col
        val = [matrix[j][i] for j in range(len(matrix))] # num rows
        transpose.append(val)
        #Can also just go transpose.append([matrix[j][i] for j in range(len(matrix))])
               
    return transpose """

""" #Option C:

def matrix_transpose(A):
    C = matrix_zeros(len(A[0]), len(A))
    for i in range(len(A[0])):
        for j in range(len(A)):
            C[i][j] = A[j][i]
    return C """

""" Question 5: Flatten Matrix -- Flattening a matrix is taking each of the rows and concatenating them into a single row.

M = [[6, 4], 
     [4, 8], 
     [3, 5]]

matrix_flatten(M)

Should return:

[6, 4, 4, 8, 3, 5] """

#Option A:

def matrix_flatten(matrix):
    ret = []
    for row in matrix:
        ret += row
    return ret

""" #Option B:

def matrix_flatten(A):
    C = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            C.append(A[i][j])
    return C """

""" #Matrix Concatenate:

def matrix_concat(A, B):

    if len(A) != len(B):
       raise ValueError('Cannot concatonate arrays with different number of rows.') 

    C = matrix_zeros(len(A), len(A[0]) + len(B[0]))
    for i, row in enumerate(A):
        for j, num in enumerate(row):
            C[i][j] = num
    for i, row in enumerate(A):
        for j, num in enumerate(row):
            C[i][j + len(A[0])] = num
    return C """

""" Run all the option A programs at once:

if __name__ == '__main__':
    print('Zeros')
    mat = matrix_zeros(4, 2)
    print_matrix(mat)
    print('Addition')
    A = [[1, 4], [2, 1]]
    B = [[6, 4], [4, 8]]
    C = matrix_add(A, B)
    print_matrix(C)
    print('Multiplication')
    A = [[1, 4, 3], 
         [2, 1, 5]]
    B = [[6, 4], 
         [4, 8], 
         [3, 5]]
    C = matrix_multiply(A, B)
    print_matrix(C)
    print('Transposition')
    A = [[6, 4],
         [4, 8],
         [3, 5]]
    B = transpose(A)
    print_matrix(B)
    print('Flatten')
    M = [[6, 4], 
         [4, 8], 
         [3, 5]]
    print(matrix_flatten(M)) """