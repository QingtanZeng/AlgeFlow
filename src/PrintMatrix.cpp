#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "PrintMatrix.hpp"

namespace AlgeFlow {


// 1. Aligned printing of colPtr, rowIdx, and values
void print_csc_aligned(const SpMatMtbl<double>& mat) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  CSC Column Pointer & Element Alignment Table" << std::endl;
    std::cout << "======================================================================" << std::endl;

    const auto& colPtr = mat.colPtr();
    const auto& rowIdx = mat.rowIdx();
    const auto& values = mat.values();

    // 1. Print colPtr array aligned with column indices
    std::cout << "\ncolPtr Array:" << std::endl;
    std::cout << "  Col Index :";
    for (size_t j = 0; j < colPtr.size(); ++j) {
        std::cout << std::setw(6) << j;
    }
    std::cout << "\n  colPtr    :";
    for (size_t j = 0; j < colPtr.size(); ++j) {
        std::cout << std::setw(6) << colPtr[j];
    }
    std::cout << "\n" << std::endl;

    // 2. Aligned column-by-column breakdown of rowIdx & values
    std::cout << " " << std::setw(6) << "Col"
              << " | " << std::setw(14) << "colPtr Range"
              << " | " << std::setw(10) << "CSC Index"
              << " | " << std::setw(10) << "Row Index"
              << " | " << std::setw(14) << "Value" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    for (size_t j = 0; j < mat.cols(); ++j) {
        size_t start = colPtr[j];
        size_t end = colPtr[j + 1];

        std::string rangeStr = "[" + std::to_string(start) + ", " + std::to_string(end) + ")";

        if (start == end) {
            std::cout << " " << std::setw(6) << j
                      << " | " << std::setw(14) << rangeStr
                      << " | " << std::setw(10) << "(empty)"
                      << " | " << std::setw(10) << "-"
                      << " | " << std::setw(14) << "-" << std::endl;
        } else {
            for (size_t p = start; p < end; ++p) {
                std::cout << " " << std::setw(6) << (p == start ? std::to_string(j) : "")
                          << " | " << std::setw(14) << (p == start ? rangeStr : "")
                          << " | " << std::setw(10) << p
                          << " | " << std::setw(10) << rowIdx[p]
                          << " | " << std::setw(14) << std::fixed << std::setprecision(4) << values[p]
                          << std::endl;
            }
        }
    }
}

// 2. Print block matrix info and mapping relationship in blkmap_
void print_spmatmtbl_info(const SpMatMtbl<double>& mat) {
    const auto& blkmapList = mat.blkmap();
    const auto& values = mat.values();

    std::cout << "Dimensions : " << mat.rows() << " x " << mat.cols() << std::endl;
    std::cout << "Non-zeros  : " << mat.nonZeros() << std::endl;
    std::cout << "Non-zeros percent : " << (double)mat.nonZeros()/(mat.rows()*mat.cols()) << std::endl;
    std::cout << "Total Sub-blocks: " << mat.getSizeBlkmap() << std::endl;

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Block Map Info & Simplified Block Matrix View (blkmap_)" << std::endl;
    std::cout << "======================================================================" << std::endl;


    // 1. High-Level Simplified Block Matrix Summary Table
    std::cout << "\n--- Simplified Block Matrix Summary ---" << std::endl;
    std::cout << " " << std::setw(10) << "Block ID"
              << " | " << std::setw(18) << "Origin (Row, Col)"
              << " | " << std::setw(16) << "Size (Rows x Cols)"
              << " | " << std::setw(14) << "Non-Zeros (NNZ)" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    for (size_t blkId = 0; blkId < blkmapList.size(); ++blkId) {
        const auto& blk = blkmapList[blkId];
        std::string posStr = "(" + std::to_string(blk.rStrt_) + ", " + std::to_string(blk.cStrt_) + ")";
        std::string sizeStr = std::to_string(blk.rows_) + " x " + std::to_string(blk.cols_);

        std::cout << " " << std::setw(10) << ("B" + std::to_string(blkId))
                  << " | " << std::setw(18) << posStr
                  << " | " << std::setw(16) << sizeStr
                  << " | " << std::setw(14) << blk.nnz_ << std::endl;
    }

    // 2. Collect unique row-start and col-start boundaries from blkmap
    std::vector<size_t> rowBnds, colBnds;
    for (const auto& blk : blkmapList) {
        rowBnds.push_back(blk.rStrt_);
        colBnds.push_back(blk.cStrt_);
    }
    std::sort(rowBnds.begin(), rowBnds.end());
    rowBnds.erase(std::unique(rowBnds.begin(), rowBnds.end()), rowBnds.end());
    std::sort(colBnds.begin(), colBnds.end());
    colBnds.erase(std::unique(colBnds.begin(), colBnds.end()), colBnds.end());

    // Build block-level 2D grid: grid[ri][ci] = blkId or -1
    size_t nBndRows = rowBnds.size();
    size_t nBndCols = colBnds.size();
    std::vector<std::vector<int>> blockGrid(nBndRows, std::vector<int>(nBndCols, -1));

    for (size_t blkId = 0; blkId < blkmapList.size(); ++blkId) {
        const auto& blk = blkmapList[blkId];
        auto ri = static_cast<size_t>(std::lower_bound(rowBnds.begin(), rowBnds.end(), blk.rStrt_) - rowBnds.begin());
        auto ci = static_cast<size_t>(std::lower_bound(colBnds.begin(), colBnds.end(), blk.cStrt_) - colBnds.begin());
        blockGrid[ri][ci] = static_cast<int>(blkId);
    }

    // 3. Simplified 2D Block Matrix Layout View (one cell per block, not per element)
    //    Compute end index of each row-band and col-band (max rStrt+rows-1 / cStrt+cols-1)
    std::vector<size_t> rowBndEnd(nBndRows, 0), colBndEnd(nBndCols, 0);
    for (const auto& blk : blkmapList) {
        auto ri = static_cast<size_t>(std::lower_bound(rowBnds.begin(), rowBnds.end(), blk.rStrt_) - rowBnds.begin());
        auto ci = static_cast<size_t>(std::lower_bound(colBnds.begin(), colBnds.end(), blk.cStrt_) - colBnds.begin());
        rowBndEnd[ri] = std::max(rowBndEnd[ri], blk.rStrt_ + blk.rows_ - 1);
        colBndEnd[ci] = std::max(colBndEnd[ci], blk.cStrt_ + blk.cols_ - 1);
    }

    // Helper: build range label, e.g. "c[0~3]"
    auto colLabel = [&](size_t ci) {
        return "c[" + std::to_string(colBnds[ci]) + "~" + std::to_string(colBndEnd[ci]) + "]";
    };
    auto rowLabel = [&](size_t ri) {
        return "r[" + std::to_string(rowBnds[ri]) + "~" + std::to_string(rowBndEnd[ri]) + "]";
    };

    // Determine column width: wide enough for the longest col-label or block label
    const int cellW = 12;
    const int rowLblW = 12;  // width for row-label column

    std::cout << "\n--- Simplified 2D Block Matrix Layout View ---" << std::endl;

    // Column header line 1: col range labels
    std::cout << std::string(rowLblW, ' ') << " |";
    for (size_t ci = 0; ci < nBndCols; ++ci) {
        std::cout << std::setw(cellW) << colLabel(ci);
    }
    std::cout << "\n" << std::string(rowLblW, '-') << "-+";
    for (size_t ci = 0; ci < nBndCols; ++ci) {
        std::cout << std::string(cellW, '-');
    }
    std::cout << std::endl;

    for (size_t ri = 0; ri < nBndRows; ++ri) {
        std::cout << std::setw(rowLblW) << rowLabel(ri) << " |";
        for (size_t ci = 0; ci < nBndCols; ++ci) {
            if (blockGrid[ri][ci] != -1) {
                std::cout << std::setw(cellW) << ("B" + std::to_string(blockGrid[ri][ci]));
            } else {
                std::cout << std::setw(cellW) << ".";
            }
        }
        std::cout << std::endl;
    }


}

void print_blkmap_mapping(const SpMatMtbl<double>& mat){

    const auto& blkmapList = mat.blkmap();
    const auto& values = mat.values();
    
    // 4. Detailed Local-to-Global Element Mappings
    std::cout << "\n--- Detailed Local-to-Global Element Mappings ---" << std::endl;
    for (size_t blkId = 0; blkId < blkmapList.size(); ++blkId) {
        const auto& blk = blkmapList[blkId];
        std::cout << "\n[Block #B" << blkId << "]"
                  << " Start: (" << blk.rStrt_ << ", " << blk.cStrt_ << ")"
                  << " | Size: " << blk.rows_ << " x " << blk.cols_
                  << " | nnz: " << blk.nnz_ << std::endl;

        std::cout << "  " << std::setw(14) << "Local (r, c)"
                  << " -> " << std::setw(14) << "World (r, c)"
                  << " -> " << std::setw(12) << "CSC Index"
                  << " -> " << std::setw(14) << "Value" << std::endl;
        std::cout << "  -------------------------------------------------------------------" << std::endl;

        for (size_t e = 0; e < blk.elemap_.size(); ++e) {
            const auto& ele = blk.elemap_[e];
            size_t wRow = blk.rStrt_ + ele.idxRLcl_;
            size_t wCol = blk.cStrt_ + ele.idxCLcl_;

            std::string lPos = "(" + std::to_string(ele.idxRLcl_) + ", " + std::to_string(ele.idxCLcl_) + ")";
            std::string wPos = "(" + std::to_string(wRow) + ", " + std::to_string(wCol) + ")";

            std::cout << "  " << std::setw(14) << lPos
                      << " -> " << std::setw(14) << wPos;

            if (ele.idxCSC_ < SIZE_MAX) {
                std::cout << " -> csc [" << std::setw(4) << ele.idxCSC_ << "]"
                          << " -> " << std::setw(14) << std::fixed << std::setprecision(4) << values[ele.idxCSC_];
            } else {
                std::cout << " -> csc [ SIZE_MAX ] (filtered/unmapped)";
            }
            std::cout << std::endl;
        }
    }
}

void print_dense_view(const SpMatMtbl<double>& mat) {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  Full Matrix Grid View" << std::endl;
    std::cout << "======================================================================" << std::endl;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());

    const auto& vals = mat.values();
    const auto& colPtr = mat.colPtr();
    const auto& rowIdx = mat.rowIdx();

    for (size_t j = 0; j < mat.cols(); ++j) {
        for (size_t p = colPtr[j]; p < colPtr[j + 1]; ++p) {
            dense(rowIdx[p], j) = vals[p];
        }
    }

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            if (std::abs(dense(i, j)) < 1e-12) {
                std::cout << std::setw(6) << ".";
            } else {
                std::cout << std::setw(6) << std::setprecision(2) << std::fixed << dense(i, j);
            }
        }
        std::cout << std::endl;
    }
}

}