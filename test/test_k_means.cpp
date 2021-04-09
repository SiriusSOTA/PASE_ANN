#include "clustering.h"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
#include "clustering.h"
#include "utils.h"
#include "pase.h"

BOOST_AUTO_TEST_SUITE(CentroidBuild)

BOOST_AUTO_TEST_CASE(JustWorks) {
    std::vector<std::vector<float>> data = {{1, 1}, {1, 2}, {3, 1}, {9, 9}, {10, 9}};
    size_t clusterCount = 2;
    size_t epochs = 5;
    float tol = 1e-4;
    IVFFlatClusterData<float> result = kMeans<float>(data, clusterCount, epochs, tol);
}

BOOST_AUTO_TEST_CASE(BuildPages) {
    const size_t centroidTuplesPerPage = 204;
    const size_t dimension = 128;
    const size_t clusterCount = 100;
    const size_t epochs = 100;
    const float tol = 1e-4;

    BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

    PaseIVFFlat<float> pase(dimension, 20);
    Parser<float> parser("../../test/test_data/siftsmall_base.fvecs", dimension, 10000);
    std::vector<std::vector<float>> parsed = parser.parse();
    IVFFlatClusterData<float> clusterData = kMeans<float>(parsed, clusterCount, epochs, tol);
}

//BOOST_AUTO_TEST_CASE(test_centroids_build_int) {
//    std::ifstream fin("../../test/test_data/clustering_in.txt", std::ios_base::in);
//    size_t num, dim, x;
//    if (fin.is_open()) {
//        fin >> num >> dim;
//        std::cout << num << " " << dim << std::endl;
//        std::vector<std::vector<int>> data(num);
//        for (size_t i = 0; i < num; ++i) {
//            for (size_t j = 0; j < dim; ++j) {
//                fin >> x;
//                data[i].push_back(x);
//            }
//        }
//        size_t clusterCount, epochs;
//        fin >> clusterCount >> epochs;
//        fin.close();
//        IVFFlatClusterData<int> result = kMeans<int>(data, clusterCount, epochs, 1e-4);
//        std::vector<int> ans(num + 1, 0);
//        std::vector<int> res(num + 1, 0);
//        std::map<int, bool> ansSize;
//        for (size_t i = 0; i < num; ++i) {
//            fin >> ans[i];
//            ansSize[ans[i]] = 1;
//        }
//        BOOST_TEST(ansSize.size() == result.idClusters.size());
//        BOOST_TEST(ansSize.size() == clusterCount);
//        for (size_t i = 0; i < result.idClusters.size(); ++i) {
//            for (size_t j = 1; j < result.idClusters[i].size(); ++j) {
//                BOOST_TEST(ans[result.idClusters[i][j]] == ans[result.idClusters[i][j - 1]]);
//            }
//        }
//    } else {
//        throw std::runtime_error("cannot open file");
//    }
//}

BOOST_AUTO_TEST_SUITE_END()