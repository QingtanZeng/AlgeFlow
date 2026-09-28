#include "SparseMatrix.hpp"

namespace AlgeFlow {

/* SPARSE MATRIX OPERATIONS PROVIDED BY THIS MODULE -------------------- */

/* 
* Sparse matrix-vector multiply for operations
* 
*		y  =  A*x  (if a > 0 && newVector == 1)
*		y +=  A*x  (if a > 0 && newVector == 0)
*		y  = -A*x  (if a < 0 && newVector == 1)
*		y -=  A*x  (if a < 0 && newVector == 0)
*
* where A is a sparse matrix and both x and y are assumed to be dense.
*/


/* 
 * Sparse matrix-transpose-vector multiply with subtraction.
 *
 * If newVector > 0, then this computes y = -A'*x,
 *                           otherwise  y -= A'*x,
 *
 * where A is a sparse matrix and both x and y are assumed to be dense.
 * If skipDiagonal == 1, then the contributions of diagonal elements are
 * not counted.
 *
 * NOTE: The product is calculating without explicitly forming the
 *       transpose.








}  // namespace AlgeFlow