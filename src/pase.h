#include "page.h"
#include "parser.h"
#include "clustering.h"
#include <functional>
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>


template<typename T>
struct PaseIVFFlat {
    size_t dimension;
    size_t clusterCount;

    CentroidPage<T> *firstCentroidPage = nullptr;
    CentroidPage<T> *lastCentroidPage = nullptr;
    typename std::vector<CentroidTuple<T>>::iterator lastCentroidElemIt;

    explicit PaseIVFFlat(size_t dimension, size_t clusterCount) : dimension(dimension), clusterCount(clusterCount) {
        if (dimension > Page<T>::calcTuplesSize()) {
            throw std::logic_error("Vector size is too big. Even one vector can not be stored on 8 KB page.");
        }
    }

    ~PaseIVFFlat() {
        auto curCentroidPage = firstCentroidPage;
        CentroidPage<T> *nextCentroidPage = nullptr;
        while (curCentroidPage->hasNextPage()) {
            nextCentroidPage = curCentroidPage->nextPage;
            delete curCentroidPage;
            curCentroidPage = nextCentroidPage;
        }
    }

private:
    void addCentroid(
            std::vector<std::reference_wrapper<std::vector<T>>> &data,
            std::vector<u_int32_t> &ids,
            std::vector<T> &centroidVector) {
        auto *firstDataPage = new DataPage<T>();
        DataPage<T> *lastDataPage = firstDataPage;
        auto lastDataElemIt = lastDataPage->tuples.begin();
        auto endTuplesIt = lastDataPage->getEndTuples(dimension);
        auto nextIdPtr = (u_int32_t *) &(*endTuplesIt);

        for (size_t i = 0; i < data.size(); ++i) {
            auto &vec = data[i];
            lastDataElemIt = std::copy(vec.get().begin(), vec.get().end(), lastDataElemIt);
            std::memcpy(nextIdPtr, &ids[i], 4);
            nextIdPtr += 4;

            if (lastDataElemIt == endTuplesIt) {
                auto newDataPage = new DataPage<T>();
                lastDataPage->nextPage = newDataPage;
                lastDataPage = newDataPage;
                lastDataElemIt = lastDataPage->tuples.begin();
                endTuplesIt = lastDataPage->getEndTuples(dimension);
                nextIdPtr = (u_int32_t *) &(*endTuplesIt);
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

public:
    void train(std::vector<std::vector<T>> &points, size_t epochs) {
        IVFFlatClusterData<T> data = kMeans(points, clusterCount, epochs);

        // all vectors have the same length
        for (u_int32_t i = 0; i < data.centroids.size(); ++i) {
            addCentroid(data.clusters[i], data.idClusters[i], data.centroids[i]);
        }
    }

    std::vector<std::vector<T>> search(const std::vector<T> &vec, size_t neighbours, size_t clusterCountToSelect) {
        using CentrWithDist = std::pair<const CentroidTuple<T> *, float>;
        using VecWithDist = std::pair<const T *, float>;

        std::vector<CentrWithDist> centrDists;

        for (CentroidPage<T> *it = firstCentroidPage; it != nullptr; it = it->nextPage) {
            for (const CentroidTuple<T> &c_tuple: it->tuples) {
                const std::vector<T> &centroid = c_tuple.vec;
                centrDists.emplace_back(&c_tuple, distance(c_tuple.vec.data(), vec.data(),
                                                           std::min(vec.size(), dimension)));
            }
        }

        //TODO: find/implement a faster way to sort
        std::sort(centrDists.begin(), centrDists.end(), [](const CentrWithDist &lhs, const CentrWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<CentroidTuple<T> *> topClusters(clusterCountToSelect);
        for (size_t i = 0; i < clusterCountToSelect; ++i) {
            topClusters[i] = centrDists[i].first;
        }

        std::vector<const VecWithDist> topVectors;

        for (CentroidTuple<T> *cluster: topClusters) {
            size_t vectorsLeft = cluster->vectorCount;
            for (DataPage<T> pg = cluster->firstDataPage; pg != nullptr; pg = pg->nextPage) {
                size_t vectorsCountOnPage = std::min(pg.calcTuplesSize() / dimension, vectorsLeft);
                vectorsLeft -= vectorsCountOnPage;
                for (size_t i = 0; i < vectorsCountOnPage; ++i) {
                    topVectors.emplace_back(pg->tuples.data() + i * dimension, 0);
                }
            }
        }

        for (VecWithDist &vDist: topVectors) {
            vDist.second = distance(vec.data(), vDist.first, dimension);
        }

        std::sort(topVectors.begin(), topVectors.end(), [](const VecWithDist &lhs, const VecWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<std::vector<T>> result(neighbours);
        for (size_t i = 0; i < neighbours; ++i) {
            result[i] = std::vector<T>(topVectors[i].first, topVectors[i].first + dimension);
        }
        return result;
    }

};



    


