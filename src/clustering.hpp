#pragma once

#include "thread_pool.hpp"
#include "calc_distance.hpp"

#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <chrono>
#include <random>
#include <set>

// debug
#include <iostream>

template<typename T>
struct IVFFlatClusterData {
    std::vector<std::vector<std::reference_wrapper<const std::vector<T>>>> clusters;
    std::vector<std::vector<u_int32_t>> idClusters;
    std::vector<std::vector<T>> centroids;

    explicit IVFFlatClusterData(size_t clustersCount) : clusters(clustersCount), idClusters(clustersCount),
                                                        centroids(clustersCount) {

    }

    // noncopyable for efficiency purposes
    IVFFlatClusterData(const IVFFlatClusterData &) = delete;

    IVFFlatClusterData &operator=(const IVFFlatClusterData &) = delete;

    IVFFlatClusterData(IVFFlatClusterData &&) noexcept = default;

    IVFFlatClusterData &operator=(IVFFlatClusterData &&) noexcept = default;
};

enum class kMeansSamplingMode {
    kNormal = 0,
    kPlusplus = 1
};

template<typename T, typename U>
inline float squaredDistance(const std::vector<T> &x, const std::vector<U> &y) {
    // ALERT: UB if x.size() > y.size()
    if (std::is_same<float, typename std::remove_cv<T>::type>::value) {
        return fvecL2sqr(&(*x.begin()), &(*y.begin()), x.size());
    }
    float squaredDist = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        squaredDist += (static_cast<float>(x[i]) - static_cast<float>(y[i])) *
                       (static_cast<float>(x[i]) - static_cast<float>(y[i]));
    }
    return squaredDist;
}

template<typename T, typename U>
float distance(const std::vector<T> &x, const std::vector<U> &y) {
    return sqrtf(squaredDistance(x, y));
}

//TODO: kmeans++ initialization support
template<typename T>
std::vector<std::vector<T>>
kMeansSample(const std::vector<std::vector<T>> &points, const size_t clusterCount,
             const kMeansSamplingMode mode) {
    if (clusterCount == 0) {
        return std::vector<std::vector<T>>();
    }
    size_t pointsCount = points.size();
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    thread_local std::mt19937 gen(seed);
    std::vector<std::vector<T>> clusters(clusterCount);
    std::set<size_t> clustersId;
    size_t index{};
    if (mode == kMeansSamplingMode::kNormal) {
        for (auto &cluster: clusters) {
            while (true) {
                index = gen() % pointsCount;
                if (clustersId.count(index) == 0) {
                    break;
                }
            }
            cluster = points[index];
            clustersId.insert(index);
        }
    } else if (mode == kMeansSamplingMode::kPlusplus) {
        // epsilon for kMeans distance
        std::vector<float> sqDistSum(pointsCount, 1e-6);

        // cycling till the end, probas ~ squared distances sum
        thread_local std::discrete_distribution distanceDistribution(sqDistSum.begin(), sqDistSum.end());
        for (auto &cluster : clusters) {
            while (true) {
                index = distanceDistribution(gen);
                if (clustersId.count(index) == 0) {
                    break;
                }
            }
            clustersId.insert(index);
            cluster = points[index];
            for (size_t j = 0; j < pointsCount; ++j) {
                sqDistSum[j] += squaredDistance(points[j], cluster);
            }
            distanceDistribution = std::discrete_distribution(sqDistSum.begin(), sqDistSum.end());
        }
    }
    return clusters;
}

template<typename T>
void assignPoints(const std::vector<std::vector<float>> &centroids, const std::vector<std::vector<T>> &points,
                  std::vector<float> &minSquaredDist,
                  std::vector<u_int32_t> &cluster) {

    auto& threadPool = getThreadPool();
    std::vector<boost::unique_future<void>> pendingTasks;
    for (size_t j = 0; j < points.size(); ++j) {

        auto calcNearestClusters = [&centroids, &minSquaredDist, &cluster, &points, j]() {
            for (size_t i = 0; i < centroids.size(); ++i) {
                // computed distance to current cluster
                float dist = squaredDistance(centroids[i], points[j]);
                // checking if distance is smaller
                if (dist < minSquaredDist[j]) {
                    minSquaredDist[j] = dist;
                    cluster[j] = i;
                }
            }
        };

        Task task(calcNearestClusters);
        boost::unique_future<void> fut = task.get_future();
        pendingTasks.push_back(std::move(fut));
        threadPool.Submit(std::move(task));
    }
    boost::wait_for_all(pendingTasks.begin(), pendingTasks.end());
}

template<typename T>
float computePoints(std::vector<std::vector<float>> &centroids, const std::vector<std::vector<T>> &points,
                    std::vector<float> &minSquaredDist,
                    std::vector<u_int32_t> &cluster) {
    // updating and computing
    auto clusterCount = static_cast<u_int32_t>(centroids.size());
    std::vector<u_int32_t> newPoints(clusterCount, 0);
    size_t dimension = points.at(0).size();
    std::vector<std::vector<float>> sum(clusterCount, std::vector<float>(dimension, 0.0));
    for (size_t j = 0; j < points.size(); ++j) {
        u_int32_t clusterId = cluster[j];
        newPoints[clusterId]++;
        for (size_t i = 0; i < points[j].size(); ++i) {
            sum[clusterId][i] += static_cast<float>(points[j][i]);
        }
        minSquaredDist[j] = std::numeric_limits<float>::max();
    }

    float frobeniusNorm = 0;
    std::vector<float> currentVec(dimension);
    for (size_t i = 0; i < centroids.size(); ++i) {
        u_int32_t clusterId = i;
        currentVec = centroids[i];
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            centroids[i][j] = sum[clusterId][j] / newPoints[clusterId];
        }
        frobeniusNorm += squaredDistance(currentVec, centroids[i]);
    }
    frobeniusNorm = sqrtf(frobeniusNorm);
    return frobeniusNorm;
}

template<typename T>
IVFFlatClusterData<T>
kMeans(const std::vector<std::vector<T>> &points, const size_t clusterCount, const size_t maxEpochs,
       const float tol) {

    std::vector<u_int32_t> pointsId(points.size(), 0);
    std::vector<float> minSquaredDist(points.size(), std::numeric_limits<float>::max());
    IVFFlatClusterData<T> data(clusterCount);

    data.centroids = kMeansSample(points, clusterCount, kMeansSamplingMode::kPlusplus);

    std::vector<std::vector<float>> centroids(data.centroids.size());
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroids[i] = std::vector<float>(data.centroids[i].begin(), data.centroids[i].end());
    }

    for (size_t i = 0; i < maxEpochs; ++i) {
        // assign cluster to points
        assignPoints(centroids, points, minSquaredDist, pointsId);
        // recompute points
        float frobeniusNorm = computePoints(centroids, points, minSquaredDist, pointsId);
//        std::cout << frobeniusNorm << std::endl;
        if (frobeniusNorm < tol) {
            std::cout << i + 1 << " epochs passed!" << std::endl;
            break;
        }
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        data.centroids[i] = std::vector<T>(centroids[i].begin(), centroids[i].end());
    }

    for (size_t i = 0; i < points.size(); ++i) {
        data.clusters[pointsId[i]].push_back(std::cref(points[i]));
        data.idClusters[pointsId[i]].push_back(i);
    }

    return std::move(data);
}

