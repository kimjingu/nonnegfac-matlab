-----------------------------------------------------
Nonnegative Matrix Factorization Algorithms Toolbox
-----------------------------------------------------
This package includes MATLAB implementations of fast numerical
algorithms for computing nonnegative matrix factorization. 
It consists of the following files.

README.txt
example_1.m
example_2.m
nmf.m
nnls1_asgivens.m
nnlsm_activeset.m
nnlsm_blockpivot.m
normalEqComb.m

Names of algorithms implemented are as follows.
'anls_bpp'     : ANLS with Block Principal Pivoting Method 
'anls_asgivens': ANLS with Active Set Method and Givens Updating
'anls_asgroup' : ANLS with Active Set Method and Column Grouping
'als'          : Alternating Least Squares Method
'hals'         : Hierarchical Alternating Least Squares Method
'mu'           : Multiplicative Updating Method

---------------
Getting Started
---------------
nmf.m is a program for executing an NMF algorithm. When A is a nonnegative 
matrix,

nmf(A,10)

returns the NMF of A with 10 as a target lower-rank. The two parameters 
(input data matrix and target lower-rank) are mandatory, whereas other 
parameters are optional. An appropriate value for the target lower-rank 
depends on each data matrix A and on the purpose of performing NMF. 
To learn optional parameters, open nmf.m and see the descriptions there. 
For example, the default algorithm for computing NMF, which is 'anls_bpp', 
can be replaced with another algorithm by specifying 'method' value as 
follows:

nmf(A,10,'method','hals')

Several usage examples are provided in example_1.m. Another example file, 
example_2.m, shows how it can be tested whether an NMF algorithm recovers 
true latent factors when applied to a synthetic matrix whose latent 
factors are known.

Default NMF algorithm is 'anls_bpp'. Another fast algorithm is 'hals'.

----------
References
----------
[1] Jingu Kim, Yunlong He, and Haesun Park.
    Algorithms for Nonnegative Matrix and Tensor Factorizations: 
    A Unified View Based on Block Coordinate Descent Framework.
    Journal of Global Optimization, 58(2), pp. 285-319, 2014.

[2] Jingu Kim and Haesun Park.
    Fast Nonnegative Matrix Factorization: An Active-set-like Method 
    And Comparisons.
    SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.

--------
Feedback
--------
Please send bug reports, comments, or questions to Jingu Kim 
(jingu.kim@gmail.com).
