#ifndef PASE_ANN_CLUSTERING_H
#define PASE_ANN_CLUSTERING_H

#include <vector>
#include <random>
#include <functional>
#include <limits>

template<typename T>
struct IVFFlatClusterData {
    std::vector<std::vector<std::reference_wrapper<std::vector<T>>>> clusters;
    std::vector<std::vector<u_int32_t>> idClusters;
    std::vector<std::vector<T>> centroids;

    explicit IVFFlatClusterData(size_t clustersCount) : clusters(clustersCount), idClusters(clustersCount),
                                                        centroids(clustersCount) {

    }

    // noncopyable on efficiency purposes
    IVFFlatClusterData(const IVFFlatClusterData &) = delete;

    IVFFlatClusterData &operator=(const IVFFlatClusterData &) = delete;
};

template<typename T>
inline float squaredDistance(const std::vector<T> &x, const std::vector<T> &y) {
    float squaredDist = 0;
    // ALERT: UB if x.size() > y.size()
    for (size_t i = 0; i < x.size(); ++i) {
        squaredDist += (static_cast<float>(x[i]) - static_cast<float>(y[i])) *
                       (static_cast<float>(x[i]) - static_cast<float>(y[i]));
    }
    return squaredDist;
}

template<typename T>
float distance(const std::vector<T> &x, const std::vector<T> &y) {
    return sqrtf(squaredDistance(x, y));
}

template<typename T>
IVFFlatClusterData<T> kMeans(std::vector<std::vector<T>> &points, size_t clusterCount, size_t maxEpochs, float tol) {
    std::mt19937 generator(0);
    size_t num = points.size();

    std::vector<u_int32_t> pointsId(points.size(), 0);
    std::vector<float> minDist(points.size(), std::numeric_limits<float>::max());
    IVFFlatClusterData<T> data(clusterCount);

    for (size_t i = 0; i < clusterCount; ++i) {
        data.centroids[i] = points[generator() % num];
    }

    while (maxEpochs--) {
        //assign cluster to points
        assignPoints(data.centroids, points, minDist, pointsId);
        //recompute points
        float frobeniusNorm = computePoints(data.centroids, points, minDist, pointsId);
        if (frobeniusNorm < tol) {
            break;
        }
    }
    for (size_t i = 0; i < points.size(); ++i) {
        data.clusters[pointsId[i]].push_back(std::ref(points[i]));
        data.idClusters[pointsId[i]].push_back(i);
    }

    return data;
}

template<typename T>
void assignPoints(std::vector<std::vector<T>> &centroids, std::vector<std::vector<T>> &points,
                  std::vector<float> &minDist,
                  std::vector<u_int32_t> &cluster) {
    for (size_t i = 0; i < centroids.size(); ++i) {
        u_int32_t clusterId = i;
        for (size_t j = 0; j < points.size(); ++j) {
            //computed distance to current cluster
            float dist = distance(centroids[i], points[j]);
            //checking if distance is smaller
            if (dist < minDist[j]) {
                minDist[j] = dist;
                cluster[j] = clusterId;
            }
        }
    }
}

template<typename T>
float computePoints(std::vector<std::vector<T>> &centroids, std::vector<std::vector<T>> &points,
                    std::vector<float> &minDist,
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
        minDist[j] = std::numeric_limits<float>::max();
    }

    float frobeniusNorm = 0;
    std::vector<T> currentVec(dimension);
    for (size_t i = 0; i < centroids.size(); ++i) {
        u_int32_t clusterId = i;
        currentVec = centroids[i];
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            //TODO: static_cast in the end
            centroids[i][j] = static_cast<T>(sum[clusterId][j] / newPoints[clusterId]);
        }
        frobeniusNorm += squaredDistance(currentVec, centroids);
    }
    frobeniusNorm = sqrtf(frobeniusNorm);
    return frobeniusNorm;
}

#endif //PASE_ANN_CLUSTERING_H
