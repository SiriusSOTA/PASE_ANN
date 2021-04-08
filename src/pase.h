#include "page.h"
#include "parser.h"
#include <functional>
#include <vector>
#include <random>
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
    private:
        void addCentroid(std::vector<std::reference_wrapper<std::vector<T>>>& data, std::vector<T>& centroidVector) {
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
    public:
        void makeCentroid(std::vector<std::vector<T>>>& points, size_t epochs) {
            kMeans(points, clusters, epochs); 
        }
};

template <typename T>
double distance(std::vector<T>& x, std::vector<T>& y) {
    size_t dimension = x.size();
    double dist = 0;
    for (size_t i = 0; i < dimension; ++i) {
        dist += (static_cast<double>(x[i]) - static_cast<double>(y[i])) * 
                (static_cast<double>(x[i]) - static_cast<double>(y[i]));
    }
    return dist;
}

template <typename T>
void kMeans(std::vector<std::vector<T>>& points, size_t clusters, size_t epochs){
    //инициализация 
    std::mt19937 generator(time(0));
    size_t num = points.size();
    size_t dim = points[0].size();
    //создание вектора центроидов
    std::vector<std::vector<T>> centroids();
    std::vector<double> minDist(points.size(), __DBL_MAX__);
    std::vector<int>& cluster(points.size(), 0);
    for (size_t i = 0; i < clusters; ++i) {
        centroids.push_back(points[generator() % num]);
    }
    while (epochs--) {
        //соотнесение точки с кластером
        assigningPoints(centroids, points, minDist, cluster);
        //пересчёт
        computingPoints(centroids, points, minDist, cluster, dim, clusters);
    }
    std::vector<std::vector<std::reference_wrapper<std::vector<T>>>> data;
    for (size_t i = 0; i < points.size(); ++i) {
        std::reference_wrapper = &points[i];
        data[cluster[i]].push_back(std::ref(points[i]>));
    }
    for (size_t i = 0; i < clusters; ++i) {
        addCentroid(data[i], cluster[i]);
    }
}


template <typename T>
void assigningPoints(std::vector<std::vector<T>>& centroids, std::vector<std::vector<T>>& points, 
                                                                    std::vector<double>& minDist, 
                                                                       std::vector<int>& cluster) {
    for (size_t i = 0; i < centroids.size(); ++i) {
        int cluster_id = i;
        for (size_t j = 0; j < points.size(); ++j) {    
            //посчитали расстояние до текущего кластера
            double dist = distance(centroids[i], points[j]);
            //проверяем расстояние
            if (dist < minDist[j]) {
                minDist[j] = dist;
                cluster[j] = cluster_id;
            }
        }
    }
}

template <typename T>
void computingPoints(std::vector<std::vector<T>>& centroids, std::vector<std::vector<T>>& points, 
                                                                    std::vector<double>& minDist, 
                                                                       std::vector<int>& cluster,
                                                                  size_t dimension, int clusters) {
    //обновление и рассчёт
    std::vector<int> newPoints(k, 0);
    std::vector<std::vector<double>> sum(clusters, std::vector<double>(dimension, 0.0));
    for (size_t j = 0; j < points.size(); ++j) {
        int clusterId = cluster[j];
        newPoints[clusterId]++;
        for (size_t i = 0; i < points[j].size(); ++i) {
            sum[cluster_id][i] += static_cast<double>(points[j][i]);
        }
        minDist[j] = __DBL_MAX__;  
    }
    //новые центроиды
    for (size_t i = 0; i < centroids.size(); ++i) {
        int cluster_id = i;
        for (size_t j = 0; j < centroids[i].size(); ++j) {
            centroids[i][j] = static_cast<T>(sum[cluster_id][j] / new_points[cluster_id]);
        }
    }
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

