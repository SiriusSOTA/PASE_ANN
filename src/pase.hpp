#pragma once

#include "page.hpp"
#include "parser.hpp"
#include "clustering.hpp"
#include "thread_pool.hpp"
#include "calc_distance.hpp"

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
        if (curCentroidPage == nullptr) {
            return;
        }
        CentroidPage<T> *nextCentroidPage = nullptr;
        while (curCentroidPage->hasNextPage()) {
            nextCentroidPage = curCentroidPage->nextPage;
            delete curCentroidPage;
            curCentroidPage = nextCentroidPage;
        }
        delete curCentroidPage;
    }

    void addCentroid(
            const std::vector<std::reference_wrapper<const std::vector<T>>> &data,
            const std::vector<u_int32_t> &ids,
            const std::vector<T> &centroidVector) {
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

    void train(const std::vector<const std::vector<T>> &points, const size_t maxEpochs, const float tol) {
        IVFFlatClusterData<T> data = kMeans(points, clusterCount, maxEpochs, tol);

        // all vectors have the same length
        for (u_int32_t i = 0; i < data.centroids.size(); ++i) {
            addCentroid(data.clusters[i], data.idClusters[i], data.centroids[i]);
        }
    }

    void add(const std::vector<std::vector<T>> &points, const std::vector<std::vector<T>> *clusters) {
        if (clusters->size() != clusterCount) {
            throw std::runtime_error("Inconsistent cluster size!");
        }
        size_t pointsCount = points.size();
        std::vector<std::vector<u_int32_t>> idClusters(clusterCount);
        for (size_t p = 0; p < pointsCount; ++p) {
            for (size_t cl = 0; cl < clusterCount; ++cl) {
                //TODO: IMPLEMENT THIS!
            }
        }
    }


    std::vector<std::vector<T>>
    findNearestVectors(const std::vector<T> &vec, const size_t neighbourCount, const size_t clusterCountToSelect) {
        auto searchResult = search(vec, neighbourCount, clusterCountToSelect);
        auto result = std::vector<std::vector<T>>();
        result.reserve(neighbourCount);

        for (auto &[vec, id]: searchResult) {
            result.push_back(std::move(vec));
        }
        return result;
    }

    std::vector<u_int32_t>
    findNearestVectorIds(const std::vector<T> &vec, const size_t neighbourCount, const size_t clusterCountToSelect) {
        auto searchResult = search(vec, neighbourCount, clusterCountToSelect);
        auto result = std::vector<u_int32_t>();
        result.reserve(neighbourCount);

        for (auto &[vec, id]: searchResult) {
            result.push_back(id);
        }
        return result;
    }

private:
    std::vector<std::pair<std::vector<T>, u_int32_t>>
    search(const std::vector<T> &vec, const size_t neighbourCount, const size_t clusterCountToSelect) {
        using CentrWithDist = std::pair<const CentroidTuple<T> *, float>;
        using VecWithDist = std::tuple<const T *, float, u_int32_t>;

        std::vector<CentrWithDist> centrDists;

        auto distanceCounter = [](const T *l, const T *r, const size_t dim) {
            if (std::is_same<float, typename std::remove_cv<T>::type>::value) {
                return fvecL2sqr(l, r,dim);
            }
            float result = 0;
            for (uint32_t i = 0; i < dim; ++i) {
                result += static_cast<float>(r[i] - l[i]) * static_cast<float>(r[i] - l[i]);
            }
            return sqrtf(result);
        };

        size_t clustersLeft = clusterCount;
        for (CentroidPage<T> *pg = firstCentroidPage; pg != nullptr; pg = pg->nextPage) {
            size_t centroidCountOnPage = std::min(pg->tuples.size(), clustersLeft);
            for (size_t i = 0; i < centroidCountOnPage; ++i) {
                const auto &centroid = pg->tuples[i];
                centrDists.emplace_back(&centroid, distanceCounter(centroid.vec.data(), vec.data(),
                                                                   std::min(vec.size(), dimension)));
            }
        }

        //TODO: find/implement a faster way to sort
        std::sort(centrDists.begin(), centrDists.end(), [](const CentrWithDist &lhs, const CentrWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<const CentroidTuple<T> *> topClusters(clusterCountToSelect);
        for (size_t i = 0; i < clusterCountToSelect; ++i) {
            topClusters[i] = centrDists[i].first;
        }

        std::vector<VecWithDist> topVectors;

        for (const CentroidTuple<T> *cluster: topClusters) {
            size_t vectorsLeft = cluster->vectorCount;
            for (const DataPage<T> *pg = cluster->firstDataPage; pg != nullptr; pg = pg->nextPage) {
                size_t vectorsCountOnPage = std::min(pg->calcVectorCount(dimension), vectorsLeft);
                auto nextIdPtr = (u_int32_t *) &(*pg->getEndTuples(dimension));
                vectorsLeft -= vectorsCountOnPage;
                for (size_t i = 0; i < vectorsCountOnPage; ++i) {
                    topVectors.emplace_back(pg->tuples.data() + i * dimension, 0, *nextIdPtr);
                    nextIdPtr += 4;
                }
            }
        }

        for (VecWithDist &vDist: topVectors) {
            std::get<1>(vDist) = distanceCounter(vec.data(), std::get<0>(vDist), dimension);
        }

        std::sort(topVectors.begin(), topVectors.end(), [](const VecWithDist &lhs, const VecWithDist &rhs) {
            return std::get<1>(lhs) < std::get<1>(rhs);
        });

        std::vector<std::pair<std::vector<T>, u_int32_t>> result(neighbourCount);
        for (size_t i = 0; i < neighbourCount; ++i) {
            result[i] = std::make_pair(
                    std::vector<T>(std::get<0>(topVectors[i]), std::get<0>(topVectors[i]) + dimension),
                    std::get<2>(topVectors[i]));
        }
        return result;
    }

};
