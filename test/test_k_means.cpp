#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <parser.h>
#include <vector>
#include "clustering.h"

BOOST_AUTO_TEST_SUITE(CentroidBuild)

BOOST_AUTO_TEST_CASE(test_centroids_build_int) {
    std::ifstream fin("../../test/test_data/clustering_in.txt"); 
    size_t num, dim, x;
    fin >> num >> dim;
    std::vector<std::vector<int>> data(num);
    for (size_t i = 0; i < num; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            fin >> x;
            data[i].push_back(x);
        }
    }
    size_t clusterCount, epochs;
    fin >> clusterCount >> epochs;
    IVFFlatClusterData<int> result = kMeans<int>(data, clusterCount, epochs, 1e-4);
}


BOOST_AUTO_TEST_SUITE_END()