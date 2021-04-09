#ifndef PASE_ANN_CLUSTERING_H
#define PASE_ANN_CLUSTERING_H

#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <chrono>

template<typename T>
struct IVFFlatClusterData {
    std::vector<std::vector<std::reference_wrapper<const std::vector<T>>>> clusters;
    std::vector<std::vector<u_int32_t>> idClusters;
    std::vector<std::vector<T>> centroids;

    explicit IVFFlatClusterData(size_t clustersCount) : clusters(clustersCount), idClusters(clustersCount),
                                                        centroids(clustersCount) {

    }

    // noncopyable on efficiency purposes
    IVFFlatClusterData(const IVFFlatClusterData &) = delete;

    IVFFlatClusterData &operator=(const IVFFlatClusterData &) = delete;

    IVFFlatClusterData(IVFFlatClusterData &&) noexcept = default;

    IVFFlatClusterData &operator=(IVFFlatClusterData &&) noexcept = default;
};

enum class kMeansSamplingMode {
    kNormal,
    kPlusplus
};

template<typename T, typename U>
inline float squaredDistance(const std::vector<T> &x, const std::vector<U> &y) {
    float squaredDist = 0;
    // ALERT: UB if x.size() > y.size()
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
kMeansSample(const std::vector<std::vector<T>> &points, size_t clusterCount, kMeansSamplingMode mode) {
    size_t pointsCount = points.size();
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::vector<std::vector<T>> clusters(clusterCount);
    if (mode == kMeansSamplingMode::kNormal) {
        for (auto &cluster: clusters) {
            cluster = points[gen() % pointsCount];
        }
    } else if (mode == kMeansSamplingMode::kPlusplus) {
        // not implemented
    }
    return clusters;
}

template<typename T>
void assignPoints(std::vector<std::vector<float>> &centroids, const std::vector<std::vector<T>> &points,
                  std::vector<float> &minSquaredDist,
                  std::vector<u_int32_t> &cluster) {
    for (size_t i = 0; i < centroids.size(); ++i) {
        u_int32_t clusterId = i;
        for (size_t j = 0; j < points.size(); ++j) {
            //computed distance to current cluster
            float dist = squaredDistance(centroids[i], points[j]);
            //checking if distance is smaller
            if (dist < minSquaredDist[j]) {
                minSquaredDist[j] = dist;
                cluster[j] = clusterId;
            }
        }
    }
}

template<typename T>
float computePoints(std::vector<std::vector<float>> &centroids, const std::vector<std::vector<T>> &points,
                    std::vector<float> &minSquaredDist,
                    std::vector<u_int32_t> &cluster) {
    // updating and computing
    auto clusterCount = static_cast<u_int32_t>(centroids.size());
    std::vector<u_int32_t> newPoints(clusterCount, 0);
    // not checking if points is empty
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

//TODO: std::cref and const reference for points - after search debugging
template<typename T>
IVFFlatClusterData<T>
kMeans(const std::vector<std::vector<T>> &points, size_t clusterCount, size_t maxEpochs, float tol) {

    std::vector<u_int32_t> pointsId(points.size(), 0);
    std::vector<float> minSquaredDist(points.size(), std::numeric_limits<float>::max());
    IVFFlatClusterData<T> data(clusterCount);

    data.centroids = kMeansSample(points, clusterCount, kMeansSamplingMode::kNormal);

    std::vector<std::vector<float>> centroids(data.centroids.size());
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroids[i] = std::vector<float>(data.centroids[i].begin(), data.centroids[i].end());
    }

    while (maxEpochs--) {
        //assign cluster to points
        assignPoints(centroids, points, minSquaredDist, pointsId);
        //recompute points
        float frobeniusNorm = computePoints(centroids, points, minSquaredDist, pointsId);
        if (frobeniusNorm < tol) {
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


#endif //PASE_ANN_CLUSTERING_H
