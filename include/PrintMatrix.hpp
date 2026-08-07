#ifndef ALGEFLOW_PRINT_MATRIX_HPP
#define ALGEFLOW_PRINT_MATRIX_HPP

#include "SparseMatrix.hpp"

namespace AlgeFlow {

void print_csc_aligned(const SpMatMtbl<double>& mat);
void print_spmatmtbl_info(const SpMatMtbl<double>& mat);
void print_blkmap_mapping(const SpMatMtbl<double>& mat);
void print_dense_view(const SpMatMtbl<double>& mat);

} // namespace AlgeFlow

#endif // ALGEFLOW_PRINT_MATRIX_HPP
