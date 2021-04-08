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
        using VecWithDist = std::pair<T*, float>;

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
                centr_dists.emplace_back(&c_tuple, distanceCounter(c_tuple.vec.data(), vec.data(),
                                                                   std::min(vec.size(), dimension)));
            }
        }

        //TODO: find/implement a faster way to sort
        std::sort(centr_dists.begin(), centr_dists.end(), [](const CentrWithDist &lhs, const CentrWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<CentroidTuple<T> *> top_clusters(n_clusters);
        for (size_t i = 0; i < n_clusters; ++i) {
            top_clusters[i] = centr_dists[i].first;
        }

        std::vector<const VecWithDist> top_vectors;

        for (CentroidTuple<T> *cluster: top_clusters) {
            size_t vectors_left = cluster->vectorCount;
            for (DataPage<T> pg = cluster->firstDataPage; pg != nullptr; pg = pg->nextPage) {
                size_t vec_on_page = std::min(pg.calcTuplesSize() / dimension, vectors_left);
                vectors_left -= vec_on_page;
                for (size_t i = 0; i < vec_on_page; ++i) {
                    top_vectors.emplace_back(pg->tuples.data() + i * dimension, 0);
                }
            }
        }

        for (VecWithDist& v_dist: top_vectors) {
            v_dist.second = distanceCounter(vec.data(), v_dist.first, dimension);
        }

        std::sort(top_vectors.begin(), top_vectors.end(), [](const VecWithDist &lhs, const VecWithDist &rhs) {
           return lhs.second < rhs.second;
        });

        std::vector<std::vector<T>> result(neighbours);
        for (size_t i = 0; i < neighbours; ++i) {
            result[i] = std::vector<T>(top_vectors[i].first, top_vectors[i].first + dimension);
        }
        return result;
    }
    
};
