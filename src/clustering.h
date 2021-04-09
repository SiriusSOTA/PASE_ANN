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
float distance(std::vector<T> &x, std::vector<T> &y) {
    float dist = 0;
    // ALERT: UB if x.size() > y.size()
    for (size_t i = 0; i < x.size(); ++i) {
        dist += (static_cast<float>(x[i]) - static_cast<float>(y[i])) *
                (static_cast<float>(x[i]) - static_cast<float>(y[i]));
    }
    return sqrtf(dist);
}

template<typename T>
IVFFlatClusterData<T> kMeans(std::vector<std::vector<T>> &points, size_t clusterCount, size_t epochs) {
    std::mt19937 generator(0);
    size_t num = points.size();
    size_t dim = points[0].size();

    std::vector<u_int32_t> pointsId(points.size(), 0);
    std::vector<float> minDist(points.size(), std::numeric_limits<float>::max());
    IVFFlatClusterData<T> data(clusterCount);

    for (size_t i = 0; i < clusterCount; ++i) {
        data.centroids[i] = points[generator() % num];
    }

    while (epochs--) {
        //assign cluster to points
        assignPoints(data.centroids, points, minDist, pointsId);
        //recompute points
        computePoints(data.centroids, points, minDist, pointsId, dim, clusterCount);
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
void computePoints(std::vector<std::vector<T>> &centroids, std::vector<std::vector<T>> &points,
                   std::vector<float> &minDist,
                   std::vector<u_int32_t> &cluster, u_int32_t clusters) {
    //updating and computing
    std::vector<u_int32_t> newPoints(clusters, 0);
    //not checking if points is empty
    size_t dimension = points.at(0).size();
    std::vector<std::vector<float>> sum(clusters, std::vector<float>(dimension, 0.0));
    for (size_t j = 0; j < points.size(); ++j) {
        u_int32_t clusterId = cluster[j];
        newPoints[clusterId]++;
        for (size_t i = 0; i < points[j].size(); ++i) {
            sum[clusterId][i] += static_cast<float>(points[j][i]);
        }
        minDist[j] = std::numeric_limits<float>::max();
    }
    //new centroids
    for (size_t i = 0; i < centroids.size(); ++i) {
        u_int32_t cluster_id = i;
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            centroids[i][j] = static_cast<T>(sum[cluster_id][j] / newPoints[cluster_id]);
        }
    }
}

#endif //PASE_ANN_CLUSTERING_H
