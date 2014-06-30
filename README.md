Nonnegative Matrix and Tensor Factorization Algorithms Toolbox
=========================
This package includes MATLAB implementations of fast optimization
algorithms for computing nonnegative matrix and tensor factorizations. 

Nonnegative Matrix Factorization (NMF)
---------------
**nmf.m** is a program for executing NMF algorithms. When A is a nonnegative
matrix,

    nmf(A,10)

returns the NMF of A with 10 as a target lower-rank. The two parameters (input
data matrix and target lower-rank) are mandatory, whereas other parameters are
optional. An appropriate value for the target lower-rank depends on each data
matrix A and on the purpose of performing NMF.  To learn optional parameters,
open **nmf.m** and see the descriptions there.  For example, the default algorithm
for computing NMF, which is `anls_bpp`, can be replaced with another algorithm
by specifying 'method' value as follows:

    nmf(A,10,'method','hals')

Names of NMF algorithms implemented are as follows.

* `anls_bpp`      -  ANLS with block principal pivoting method
* `anls_asgivens` -  ANLS with active set method and givens updating
* `anls_asgroup`  -  ANLS with active set method and column grouping
* `als`           -  Alternating least squares method
* `hals`          -  Hierarchical alternating least squares method
* `mu`            -  Multiplicative updating method

Several usage examples are provided in **example_nmf_1.m**. Another example file,
**example_nmf_2.m**, shows how it can be tested whether an NMF algorithm recovers
true latent factors when applied to a synthetic matrix whose latent
factors are known.

Default NMF algorithm is `anls_bpp`. Another fast algorithm is `hals`.

Nonnegative Tensor Factorization (Nonnegative CP)
---------------
This software performs nonnegative tensor factorization in CP (Canonical
Decomposition / PARAFAC) model. **ncp.m** is a program for executing NTF
algorithms.

To use this program, it is necessary to first install MATLAB Tensor Toolbox by
Brett W. Bader and Tamara G. Kolda, available at
http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox/. The latest version that was
tested with this program is Version 2.4, March 2010. Refer to the help manual
of the toolbox for installation and basic usage.

Please see the description in **ncp.m** and try to execute **example_ncp.m**
to learn how to use this program.  Names of nonnegative CP algorithms
implemented are as follows.

* `anls_bpp`     - ANLS with block principal pivoting method
* `anls_asgroup` - ANLS with active set method and column grouping
* `hals`         - Hierarchical alternating least squares method
* `mu`           - Multiplicative updating method

References
----------
1. Jingu Kim, Yunlong He, and Haesun Park.
   Algorithms for Nonnegative Matrix and Tensor Factorizations:
   A Unified View Based on Block Coordinate Descent Framework.
   Journal of Global Optimization, 58(2), pp. 285-319, 2014.

2. Jingu Kim and Haesun Park.
   Fast Nonnegative Matrix Factorization: An Active-set-like Method
   And Comparisons.
   SIAM Journal on Scientific Computing (SISC), 33(6), pp. 3261-3281, 2011.

3. Jingu Kim and Haesun Park.
   Fast Nonnegative Tensor Factorization with an Active-set-like Method.
   In High-Performance Scientific Computing: Algorithms and Applications, 
   Springer, 2012, pp. 311-326.

Feedback
--------

Please send bug reports, comments, or questions to [Jingu Kim](mailto:jingu.kim@gmail.com).
Contributions and extentions with new algorithms are welcome.
