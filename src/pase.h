#include "page.h"
#include "parser.h"
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
    float distance(std::vector<T> &x, std::vector<T> &y) {
        float dist = 0;
        for (size_t i = 0; i < dimension; ++i) {
            dist += (static_cast<float>(x[i]) - static_cast<float>(y[i])) *
                    (static_cast<float>(x[i]) - static_cast<float>(y[i]));
        }
        return sqrtf(dist);
    }

    void kMeans(std::vector<std::vector<T>> &points, size_t clusterCount, size_t epochs) {
        //инициализация 
        std::mt19937 generator(0);
        size_t num = points.size();
        size_t dim = points[0].size();
        //создание вектора центроидов
        std::vector<std::vector<T>> centroids;
        std::vector<float> minDist(points.size(), __FLT_MAX__);
        std::vector<int> &cluster(points.size(), 0);
        for (size_t i = 0; i < clusterCount; ++i) {
            centroids.push_back(points[generator() % num]);
        }
        while (epochs--) {
            //соотнесение точки с кластером
            assigningPoints(centroids, points, minDist, cluster);
            //пересчёт
            computingPoints(centroids, points, minDist, cluster, dim, clusterCount);
        }
        std::vector<std::vector<std::reference_wrapper<std::vector<T>>>> data(clusterCount);
        std::vector<std::vector<u_int32_t>> ids(clusterCount);
        for (size_t i = 0; i < points.size(); ++i) {
            data[cluster[i]].push_back(std::ref(points[i]));
            ids[cluster[i]].push_back(i);
        }
        for (size_t i = 0; i < clusterCount; ++i) {
            addCentroid(data[i], ids[i], cluster[i]);
        }
    }


    void assigningPoints(std::vector<std::vector<T>> &centroids, std::vector<std::vector<T>> &points,
                         std::vector<float> &minDist,
                         std::vector<int> &cluster) {
        for (size_t i = 0; i < centroids.size(); ++i) {
            int cluster_id = i;
            for (size_t j = 0; j < points.size(); ++j) {
                //посчитали расстояние до текущего кластера
                float dist = distance(centroids[i], points[j]);
                //проверяем расстояние
                if (dist < minDist[j]) {
                    minDist[j] = dist;
                    cluster[j] = cluster_id;
                }
            }
        }
    }

public:
    void buildIndex(std::vector<std::vector<T>> &points, size_t epochs) {
        kMeans(points, clusterCount, epochs);
    }

    void addCentroid(
            std::vector<std::reference_wrapper<std::vector<T>>> &data,
            std::vector<u_int32_t> &ids,
            std::vector<T> &centroidVector) {
        auto *firstDataPage = new DataPage<T>();
        DataPage<T> *lastDataPage = firstDataPage;
        auto lastDataElemIt = lastDataPage->tuples.begin();
        auto endTuplesIt = lastDataPage->getEndTuples(dimension);
        auto nextIdPtr = &(*endTuplesIt);

        for (size_t i = 0; i < data.size(); ++i) {
            auto& vec = data[i];
            lastDataElemIt = std::copy(vec.get().begin(), vec.get().end(), lastDataElemIt);
            std::memcpy(nextIdPtr, &ids[i], 4);
            nextIdPtr += 4;

            if (lastDataElemIt == endTuplesIt) {
                auto newDataPage = new DataPage<T>();
                lastDataPage->nextPage = newDataPage;
                lastDataPage = newDataPage;
                lastDataElemIt = lastDataPage->tuples.begin();
                endTuplesIt = lastDataPage->getEndTuples(dimension);
                nextIdPtr = &(*endTuplesIt);
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

    void computingPoints(std::vector<std::vector<T>> &centroids, std::vector<std::vector<T>> &points,
                         std::vector<float> &minDist,
                         std::vector<int> &cluster, int clusters) {
        //обновление и рассчёт
        std::vector<int> newPoints(clusters, 0);
        std::vector<std::vector<float>> sum(clusters, std::vector<float>(dimension, 0.0));
        for (size_t j = 0; j < points.size(); ++j) {
            int clusterId = cluster[j];
            newPoints[clusterId]++;
            for (size_t i = 0; i < points[j].size(); ++i) {
                sum[clusterId][i] += static_cast<float>(points[j][i]);
            }
            minDist[j] = __FLT_MAX__;
        }
        //новые центроиды
        for (size_t i = 0; i < centroids.size(); ++i) {
            int cluster_id = i;
            for (size_t j = 0; j < centroids[i].size(); ++j) {
                centroids[i][j] = static_cast<T>(sum[cluster_id][j] / newPoints[cluster_id]);
            }
        }
    }

    void search(const std::vector<T> &vec, size_t neighbours, size_t clusterCountToSelect) {
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



    


