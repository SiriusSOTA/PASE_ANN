#include "page.h"
#include <vector>
#include <stdexcept>
#include <algorithm>

template<typename T>
struct PaseIVFFlat {
    size_t dimension;
    CentroidPage<T> *firstCentroidPage = nullptr;
    CentroidPage<T> *lastCentroidPage = nullptr;
    typename std::vector<CentroidTuple<T>>::iterator lastCentroidElemIt;

    explicit PaseIVFFlat(size_t dimension) : dimension(dimension) {
        if (dimension > Page<T>::calcTuplesSize()) {
            throw std::logic_error("Vector size is too big. Even one vector can not be stored on 8 KB page.");
        }
    }

    void addCentroid(std::vector<std::vector<T>> &data, std::vector<T> &centroidVector) {
        auto *firstDataPage = new DataPage<T>();
        DataPage<T> *lastDataPage = firstDataPage;
        auto lastDataElemIt = lastDataPage->tuples.begin();

        for (auto &vec: data) {
            lastDataElemIt = std::copy(vec.begin(), vec.end(), lastDataElemIt);
            if (lastDataElemIt > lastDataPage->tuples.end() - dimension) {
                auto newDataPage = new DataPage<T>();
                lastDataPage->nextPage = newDataPage;
                lastDataPage = newDataPage;
                lastDataElemIt = lastDataPage->tuples.begin();
            }
        }

        if (!firstCentroidPage) {
            firstCentroidPage = new CentroidPage<T>();
            lastCentroidPage = firstCentroidPage;
            lastCentroidElemIt = firstCentroidPage->tuples.begin();
        }

        if (lastCentroidElemIt == lastCentroidPage->tuples.end()) {
            auto newCentroidPage = new CentroidPage<T>();
            lastCentroidPage->nextPage = newCentroidPage;
            lastCentroidPage = newCentroidPage;
            lastCentroidElemIt = lastCentroidPage->tuples.begin();
        }
        lastCentroidElemIt->vec = centroidVector;
        lastCentroidElemIt->vectorCount = data.size();
        lastCentroidElemIt->firstDataPage = firstDataPage;
        lastCentroidElemIt += 1;
    }

    void search(const std::vector<T> &vec, size_t neighbours, size_t n_clusters) {
        using CentrWithDist = std::pair<const CentroidTuple<T> *, float>;

        std::vector<CentrWithDist> centr_dists;

        auto distanceCounter = [](T *l, T *r, size_t size) {
            float result = 0;
            for (uint32_t i = 0; i < size; ++i) {
                result += (r[i] - l[i]) * (r[i] - l[i]);
            }
            return result;
        };
        for (CentroidPage<T> *it = firstCentroidPage; it != nullptr; it = it->nextPage) {
            for (const CentroidTuple<T> &c_tuple: it->tuples) {
                const std::vector<T> &centroid = c_tuple.vec;
                //min may be not the best option for sorting size, but it won't crash
                centr_dists.emplace_back(&c_tuple, distanceCounter(c_tuple.vec.data(), vec.data(),
                                                                   std::min(vec.size(), c_tuple.vec.size())));
            }
        }

        //TODO: find/implement a faster way to sort
        std::sort(centr_dists.begin(), centr_dists.end(), [](const CentrWithDist &lhs, const CentrWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<CentroidTuple<T>> top_clusters;
        for (auto cl_pair: centr_dists) {
            //TODO: take top_k centroid_tuple
        }

        for (CentroidPage<T> *it = firstCentroidPage; it != nullptr; it = it->nextPage) {
            for (const CentroidTuple<T> &c_tuple: it->tuples) {
                const std::vector<T> &centroid = c_tuple.vec;
                for (DataPage<T> *pg = c_tuple.firstDataPage; pg != nullptr; pg = pg->nextPage) {

                }
            }
        }
    }


};
