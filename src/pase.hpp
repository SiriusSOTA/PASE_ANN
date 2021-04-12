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

    void train(const std::vector<std::vector<T>> &points, const size_t maxEpochs, const float tol) {
        IVFFlatClusterData<T> data = kMeans(points, clusterCount, maxEpochs, tol);

        // all vectors have the same length
        for (u_int32_t i = 0; i < data.centroids.size(); ++i) {
            addCentroid(data.clusters[i], data.idClusters[i], data.centroids[i]);
        }
    }

    void add(const std::vector<std::vector<T>> &points) {

        size_t pointsCount = points.size();
        std::vector<std::vector<u_int32_t>> idClusters(clusterCount);
        std::vector<float> tempDistances(clusterCount);
        std::vector<uint32_t> clusterIndexes(pointsCount);
        std::vector<boost::unique_future<void>> pendingTasks;
        auto &threadPool = getThreadPool();

        for (size_t pId = 0; pId < pointsCount; ++pId) {
            const std::vector<T> &point = points[pId];
            auto findClosestCentroid = [this, &point, pId, &clusterIndexes]() {
                size_t clustersLeft = clusterCount;
                std::vector<float> currDistances(clusterCount);

                for (CentroidPage<T> *pg = firstCentroidPage; pg != nullptr; pg = pg->nextPage) {
                    size_t centroidCountOnPage = std::min(pg->tuples.size(), clustersLeft);

                    for (size_t i = 0; i < centroidCountOnPage; ++i) {
                        const auto &centroid = pg->tuples[i];
                        currDistances[clusterCount - clustersLeft + i] = distanceCounter(centroid.vec.data(),
                                                                                         point.data(), dimension);
                    }
                    clustersLeft -= centroidCountOnPage;
                }

                size_t minClusterIndex = std::distance(currDistances.begin(),
                                                       std::min_element(currDistances.begin(), currDistances.end()));
                clusterIndexes[pId] = minClusterIndex;
            };

            Task task(findClosestCentroid);
            boost::unique_future<void> fut = task.get_future();
            pendingTasks.push_back(std::move(fut));
            threadPool.Submit(std::move(task));
        }
        boost::wait_for_all(pendingTasks.begin(), pendingTasks.end());
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

        std::vector<CentrWithDist> centrDists(clusterCount);

        auto &threadPool = getThreadPool();
        std::vector<boost::unique_future<void>> pendingTasks;

        size_t clustersLeft = clusterCount;
        for (CentroidPage<T> *pg = firstCentroidPage; pg != nullptr; pg = pg->nextPage) {
            size_t centroidCountOnPage = std::min(pg->tuples.size(), clustersLeft);

            auto calcDistsToCentroids = [this, &centrDists, &vec, centroidCountOnPage, clustersLeft, pg]() {
                for (size_t i = 0; i < centroidCountOnPage; ++i) {
                    const auto &centroid = pg->tuples[i];
                    centrDists[clusterCount - clustersLeft + i] = std::move(
                            CentrWithDist(&centroid, distanceCounter(centroid.vec.data(), vec.data(),
                                                                     std::min(vec.size(), dimension))));
                }
            };
            Task task(calcDistsToCentroids);
            boost::unique_future<void> fut = task.get_future();
            pendingTasks.push_back(std::move(fut));
            threadPool.Submit(std::move(task));

            clustersLeft -= centroidCountOnPage;
        }
        boost::wait_for_all(pendingTasks.begin(), pendingTasks.end());


        //TODO: find/implement a faster way to sort
        std::sort(centrDists.begin(), centrDists.end(), [](const CentrWithDist &lhs, const CentrWithDist &rhs) {
            return lhs.second < rhs.second;
        });

        std::vector<const CentroidTuple<T> *> topClusters(clusterCountToSelect);
        for (size_t i = 0; i < clusterCountToSelect; ++i) {
            topClusters[i] = centrDists[i].first;
        }

        size_t vectorCount = 0;
        for (const CentroidTuple<T> *cluster: topClusters) {
            vectorCount += cluster->vectorCount;
        }

        std::vector<VecWithDist> topVectors(vectorCount);
        size_t topVectorIdx = 0;

        for (const CentroidTuple<T> *cluster: topClusters) {
            auto getTopVectors = [this, cluster, topVectorIdx, &topVectors]() {
                auto vectorIdx = topVectorIdx;
                size_t vectorsLeft = cluster->vectorCount;
                for (const DataPage<T> *pg = cluster->firstDataPage; pg != nullptr; pg = pg->nextPage) {
                    size_t vectorsCountOnPage = std::min(pg->calcVectorCount(dimension), vectorsLeft);
                    auto nextIdPtr = (u_int32_t *) &(*pg->getEndTuples(dimension));
                    vectorsLeft -= vectorsCountOnPage;
                    for (size_t i = 0; i < vectorsCountOnPage; ++i) {
                        topVectors[vectorIdx] = std::tuple(pg->tuples.data() + i * dimension, 0, *nextIdPtr);
                        nextIdPtr += 4;
                        ++vectorIdx;
                    }
                }
            };
            Task task(getTopVectors);
            boost::unique_future<void> fut = task.get_future();
            pendingTasks.push_back(std::move(fut));
            threadPool.Submit(std::move(task));

            topVectorIdx += cluster->vectorCount;
        }
        boost::wait_for_all(pendingTasks.begin(), pendingTasks.end());

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

private:
    float distanceCounter(const T *l, const T *r, const size_t dim) {
        if (std::is_same<float, typename std::remove_cv<T>::type>::value) {
            return fvecL2sqr(l, r, dim);
        }
        float result = 0;
        for (uint32_t i = 0; i < dim; ++i) {
            result += static_cast<float>(r[i] - l[i]) * static_cast<float>(r[i] - l[i]);
        }
        return sqrtf(result);
    }
};

