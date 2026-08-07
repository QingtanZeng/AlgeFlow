#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdint.h>
#include <array>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <utility>
#include <cmath>

#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/Dense"

namespace AlgeFlow {

/* define a static structure sparse matrix from SpMatMttr.setFromTriplets precomputaion in compile-time using CRC*/
/*Only used in read/write and calculation in runtime, rather than block matrix concatenation */
template<typename T, size_t Rows, size_t Cols, size_t NNZ, size_t numBlk >
class SpMatSttc{
public:
    size_t m_ = Rows;     // Row
    size_t n_ = Cols;     // Column
    size_t nnz_ = NNZ;   // non-zero values

    /*nested datatype*/
    struct BlkMap{
        size_t rStrt_, cStrt_;        // starting coordinates in full matrix
        size_t rows_, cols_;          // size of block matrix
        size_t nnz_;                    // nnz

        size_t offsetelemap_;           // from offset to offset+nnz in elemap_
    };
    struct EleMap{
        size_t idxRLcl_, idxCLcl_;    // local coordinate of elements in block matrix
        size_t idxCSC_;
    };
private:
    std::array<T, NNZ> values_;             // value data
    std::array<size_t, NNZ> rowIdx_;      // Raw index
    std::array<size_t, Cols+1> colPtr_;   // Column index

    std::array<BlkMap, numBlk> blkmap_;     // block matrix's map
    std::array<EleMap, NNZ>    elemap_;     // elemap by local CSC index
};

/*define mutable structure sparse matrix for precomputaion in compile-time using COO */
/*Only used in block matrix concatenation, rather than read/write and calculation*/
template<typename T>
class SpMatMtbl{
public:
    /*Enum*/
    enum class ConcatMode:uint8_t{Rght, Bttm, BttmRght, TopRght, BttmLft};
    enum class DrctShift:uint8_t{Rght, Down};
    /*nested datatype*/
    struct Triplet{
        size_t row, col;
        T value;
        bool isHld = false;
        size_t blkId = SIZE_MAX;
        size_t idxCSCLcl = SIZE_MAX;       // local CSC index in block matrix

        Triplet(size_t r, size_t c, T v, bool hld=true, size_t bId=SIZE_MAX, size_t IdxLcl=SIZE_MAX) 
                    : row(r), col(c), value(v), isHld(hld), blkId(bId), idxCSCLcl(IdxLcl){}
    };
    struct BlkMap{
        size_t rStrt_, cStrt_;        // starting coordinates in full matrix
        size_t rows_, cols_;          // size of block matrix
        size_t nnz_;                 // nnz

        struct EleMap{
            size_t idxRLcl_, idxCLcl_;    // local coordinate of elements in block matrix
            size_t idxCSC_;           // wolrd coordinate in full matrix
        };
        std::vector<EleMap> elemap_;        // elemap by local CSC index

        BlkMap(size_t rS=0, size_t cS=0, size_t r=0, size_t c=0, size_t nnz=0) 
                : rStrt_(rS), cStrt_(cS), rows_(r), cols_(c), nnz_(nnz){
            elemap_.reserve(nnz_);
        }
    };

    /*public variables*/
    size_t m_ ;     // Row
    size_t n_ ;     // Column
    // size_t nnz_ = 0;   // non-zero values, only used in SpMatSttc
    /*Constructor & Destructor*/
    SpMatMtbl(size_t r=0, size_t c=0) : m_(r), n_(c), isSrt_(false){}
    ~SpMatMtbl() = default;

    /*Read Class states or data*/
    size_t getSizeTriplets() const { return triplets_.size(); } 
    size_t getSizeBlkmap() const { return blkmap_.size(); } 
    size_t rows() const { return m_; }
    size_t cols() const { return n_; }
    size_t nonZeros() const { return values_.size(); }
    const std::vector<T>& values() const { return values_; }
    const std::vector<size_t>& colPtr() const { return colPtr_; }
    const std::vector<size_t>& rowIdx() const { return rowIdx_; }

    /*Function(Not Used): Add separate element */
    void addElem(size_t r, size_t c, T v){
        if (r>= m_ || c>= n_) throw std::out_of_range("Index out of bounds");
        triplets_.emplace_back(r, c, v, true);
        isSrt_ = false;
    }
    /*Function: Add block matrix through dense or sparse matrix*/
    template<typename Derived>
    void addBlkMtrx(size_t rStrt, size_t cStrt, const Eigen::MatrixBase<Derived>& mat, bool isSp, double epsilon=1e-9){
        static_assert(std::is_same<typename Derived::Scalar, T>::value, "Scalar type of the block matrix must match SpMatMtbl's T");
        size_t mRows = static_cast<size_t>(mat.rows());
        size_t nCols = static_cast<size_t>(mat.cols());
        if( (rStrt + mRows > m_) || (cStrt + nCols > n_) ){
            throw std::out_of_range("Block matrix exceeds outer matrix bounds");
        }
        //register a new mapping 
        size_t blkId = blkmap_.size();
        blkmap_.emplace_back(rStrt, cStrt, mRows, nCols, mRows * nCols);
        auto& blkmap = blkmap_.back();

        if(isSp==false)
            triplets_.reserve(triplets_.size() + mRows * nCols);
        else
            triplets_.reserve(triplets_.size() + (size_t)(0.1 * mRows * nCols) );
        // LOOP
        size_t idxCSCLcl = 0;
        for(size_t idxC = 0; idxC != nCols; ++idxC){ // Column Major
            for (size_t idxR=0; idxR!=mRows; ++idxR) {
                bool flgKeep = !isSp || (std::abs(mat(idxR, idxC)) > epsilon) ;
                if(flgKeep){
                    triplets_.emplace_back(rStrt+idxR, cStrt+idxC, mat(idxR, idxC), true, blkId, idxCSCLcl);
                    blkmap.elemap_.emplace_back(typename BlkMap::EleMap{idxR, idxC, SIZE_MAX});
                    ++idxCSCLcl;
                }
            }
        }
        isSrt_ = false;
    }
    template<typename Derived>
    void addBlkMtrx(size_t rStrt, size_t cStrt, const Eigen::SparseMatrixBase<Derived>& smat){
        static_assert(std::is_same<typename Derived::Scalar, T>::value, "Scalar type of the block matrix must match SpMatMtbl's T");
        size_t mRows = static_cast<size_t>(smat.rows());
        size_t nCols = static_cast<size_t>(smat.cols());
        size_t nnz = static_cast<size_t>(smat.derived().nonZeros());
        if( (rStrt + mRows > m_) || (cStrt + nCols > n_) ){
            throw std::out_of_range("Block matrix exceeds outer matrix bounds");
        }

        //register a new mapping 
        size_t blkId = blkmap_.size();
        blkmap_.emplace_back(rStrt, cStrt, mRows, nCols, nnz);
        auto& blkmap = blkmap_.back();

        triplets_.reserve(triplets_.size() + nnz);
        // LOOP
        size_t idxCSCLcl = 0;
        for(size_t k=0; k!=static_cast<size_t>(smat.outerSize()); ++k){
            for(typename Derived::InnerIterator it(smat.derived(), k); it; ++it){
                size_t rLcl = static_cast<size_t>(it.row());
                size_t cLcl = static_cast<size_t>(it.col());
                triplets_.emplace_back(rStrt + rLcl, cStrt + cLcl, it.value(), true, blkId, idxCSCLcl);
                blkmap.elemap_.emplace_back(typename BlkMap::EleMap{rLcl, cLcl, SIZE_MAX});
                ++idxCSCLcl;
            }
        }
        isSrt_ = false;
    }

    /*Function: Stack block matrix through dense or sparse matrix or SpMatMtbl*/
    template<typename Derived>
    void stack(const Eigen::MatrixBase<Derived>& mat, ConcatMode catmode, bool isSp, double epsilon=1e-9){
        static_assert(std::is_same<typename Derived::Scalar, T>::value, "Scalar type of the block matrix must match SpMatMtbl's T");
        size_t mRows = static_cast<size_t>(mat.rows());
        size_t nCols = static_cast<size_t>(mat.cols());
        
        // move self matrix and determine concated matrix position
        auto [rStrt, cStrt] = ConcatMove(catmode, mRows, nCols);

        //register a new mapping 
        size_t blkId = blkmap_.size();
        blkmap_.emplace_back(rStrt, cStrt, mRows, nCols, mRows * nCols);
        auto& blkmap = blkmap_.back();

        if(isSp==false)
            triplets_.reserve(triplets_.size() + mRows * nCols);
        else
            triplets_.reserve(triplets_.size() + (size_t)(0.1 * mRows * nCols) );
        // LOOP
        size_t idxCSCLcl = 0;
        for(size_t idxC = 0; idxC != nCols; ++idxC){ // Column Major
            for (size_t idxR=0; idxR!=mRows; ++idxR) {
                bool flgKeep = !isSp || (std::abs(mat(idxR, idxC)) > epsilon) ;
                if(flgKeep){
                    triplets_.emplace_back(rStrt+idxR, cStrt+idxC, mat(idxR, idxC), true, blkId, idxCSCLcl);
                    blkmap.elemap_.emplace_back(typename BlkMap::EleMap{idxR, idxC, SIZE_MAX});
                    ++idxCSCLcl;
                }
            }
        }
        isSrt_ = false;
    }
    template<typename Derived>
    void stack(const Eigen::SparseMatrixBase<Derived>& smat, ConcatMode catmode){
        static_assert(std::is_same<typename Derived::Scalar, T>::value, "Scalar type of the block matrix must match SpMatMtbl's T");
        size_t mRows = static_cast<size_t>(smat.rows());
        size_t nCols = static_cast<size_t>(smat.cols());
        size_t nnz = static_cast<size_t>(smat.derived().nonZeros());
        
        // move self matrix and determine concated matrix position
        auto [rStrt, cStrt] = ConcatMove(catmode, mRows, nCols);

        //register a new mapping 
        size_t blkId = blkmap_.size();
        blkmap_.emplace_back(rStrt, cStrt, mRows, nCols, nnz);
        auto& blkmap = blkmap_.back();

        triplets_.reserve(triplets_.size() + nnz);
        // LOOP
        size_t idxCSCLcl = 0;
        for(size_t k=0; k!=static_cast<size_t>(smat.outerSize()); ++k){
            for(typename Derived::InnerIterator it(smat.derived(), k); it; ++it){
                size_t rLcl = static_cast<size_t>(it.row());
                size_t cLcl = static_cast<size_t>(it.col());
                triplets_.emplace_back(rStrt + rLcl, cStrt + cLcl, it.value(), true, blkId, idxCSCLcl);
                blkmap.elemap_.emplace_back(typename BlkMap::EleMap{rLcl, cLcl, SIZE_MAX});
                ++idxCSCLcl;
            }
        }
        isSrt_ = false;
    }

    void stack(SpMatMtbl<T>&& mat, ConcatMode catmode){
        // move self matrix and determine concated matrix position
        auto [rStrt, cStrt] = ConcatMove(catmode, mat.m_, mat.n_);
        // vector dilation
        this->triplets_.reserve(this->triplets_.size() + mat.getSizeTriplets());
        this->blkmap_.reserve(this->blkmap_.size() + mat.getSizeBlkmap());

        size_t offsetBlkId = this->blkmap_.size();      // append B.blkId at end of A
        for(auto& blk: mat.blkmap_){
            blk.rStrt_ += rStrt ;
            blk.cStrt_ += cStrt ;
            this->blkmap_.emplace_back(std::move(blk));
        }

        // copy/move into self
        for(const auto& trip: mat.triplets_){
            this->triplets_.emplace_back(trip.row + rStrt, trip.col + cStrt, std::move(trip.value), trip.isHld,
                    trip.blkId + offsetBlkId, trip.idxCSCLcl);
        }
        // clear input concated matrix
        // mat.~SpMatMtbl();    //BUG FIX: 坚决避免对右值引用显式调用析构函数，会导致 double free
        mat.triplets_.clear();
        mat.blkmap_.clear();
        
        isSrt_ = false;
    }

    /*Function: sorted and compressed to CSC*/
    void setFromTriplets(double epsilon=1e-9){
        // 1. 按照 CSC 的主序进行排序：先按列排，列相同则按行排
        std::sort(triplets_.begin(), triplets_.end(),[](const Triplet& a, const Triplet& b){
            if(a.col!=b.col) return a.col<b.col;
            return a.row<b.row;
        });

        // reset vector
        values_.clear();
        rowIdx_.clear();
        if (triplets_.empty()) return;
        values_.reserve(triplets_.size());
        rowIdx_.reserve(triplets_.size());
        colPtr_.assign(n_+1, 0);

        // 2. Traverse and generate CSC by Columns > Row
        size_t curCol = 0;
        for(const auto& elem : triplets_){
            if(elem.row>=m_ || elem.col>=n_) throw std::out_of_range("Element' index out of matrix bounds");

            // world CSC index
            size_t idxCSC = SIZE_MAX;
            
            // check whether into new column, or update colPtr_: middle columns might be empty
            while(elem.col != curCol){
                ++curCol;
                colPtr_[curCol] = values_.size();
                // If Col0 has values, colPtr_[0]==0; If not, colPtr_[0]==colPtr_[1]=0 either.
                // If Colj is empty, colPtr_[j]==colPtr_[j+1]
            }
            // add new element or accumulation same element
            if(!values_.empty() && rowIdx_.back()==elem.row && curCol==elem.col){
                values_.back() += elem.value;
                idxCSC = values_.size()-1;
            }else{
                if(elem.isHld==true || std::abs(elem.value)>epsilon ){ 
                    rowIdx_.push_back(elem.row);
                    values_.push_back(elem.value);
                    idxCSC = values_.size()-1;
                }else{
                    idxCSC = SIZE_MAX ;
                }
            }

            // write world CSC index in full matrix
            if(elem.blkId!=SIZE_MAX && idxCSC != SIZE_MAX){
                blkmap_[elem.blkId].elemap_[elem.idxCSCLcl].idxCSC_ = idxCSC;
            }

        }

        // 3. fill in tail of colPtr_
        while(curCol != n_){
           ++curCol;
           colPtr_[curCol] = values_.size();
        }
        isSrt_ = true;
    }

private:
    mutable std::vector<Triplet> triplets_;
    mutable std::vector<BlkMap> blkmap_;
    mutable bool isSrt_ = false;

    std::vector<T> values_;
    std::vector<size_t> colPtr_;
    std::vector<size_t> rowIdx_;

    /*Internal auxiliary function*/
    std::pair<size_t, size_t> 
    ConcatMove(ConcatMode catmode, size_t mRows, size_t nCols){
        size_t rStrt = 0, cStrt = 0;    // Start of input block matrix
        size_t old_m = this->m_;
        size_t old_n = this->n_;

        switch (catmode) {
            case ConcatMode::Rght :
                rStrt = 0;
                cStrt = old_n;
                this->n_ += nCols;
                this->m_ = std::max(old_m, mRows);
                break;
            case ConcatMode::Bttm :
                rStrt = old_m;
                cStrt = 0;
                this->m_ += mRows;
                this->n_ = std::max(old_n, nCols);
                break;
            case ConcatMode::BttmRght :
                rStrt = old_m;
                cStrt = old_n;
                this->m_ += mRows;
                this->n_ += nCols;
                break;
            case ConcatMode::TopRght :      // Special case, first move self down
                shift(mRows, DrctShift::Down);
                rStrt = 0;
                cStrt = old_n;
                this->m_ += mRows;
                this->n_ += nCols;
                break;
            case ConcatMode::BttmLft :      // Special case, first move self right
                shift(nCols, DrctShift::Rght);
                rStrt = old_m;
                cStrt = 0;
                this->m_ += mRows;
                this->n_ += nCols;
                break;
        }

        return std::make_pair(rStrt, cStrt);
    }
    void shift(size_t N, DrctShift drct){
        if(N==0) return;
        switch (drct) {
            case DrctShift::Rght :
                // move block matrix
                for(auto& blk: blkmap_) blk.cStrt_ += N;
                // move triplet
                for(auto& trip: triplets_) trip.col += N;
                break;
            case DrctShift::Down :
                // move block matrix
                for(auto& blk: blkmap_) blk.rStrt_ += N;
                // move triplet
                for(auto& trip: triplets_) trip.row += N;
                break;
        }
    }
};

} // namespace AlgeFlow
