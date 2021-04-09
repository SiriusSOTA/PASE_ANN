#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <parser.h>
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
    const size_t clusterCount = 30;
    const size_t epochs = 100;
    const float tol = 1e-4;

    BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

    PaseIVFFlat<float> pase(dimension, 20);
    Parser<float> parser("../../test/test_data/siftsmall_base.fvecs", dimension, 10000);
    std::vector<std::vector<float>> parsed = parser.parse();
    IVFFlatClusterData<float> clusterData = kMeans<float>(parsed, clusterCount, epochs, tol);
}

BOOST_AUTO_TEST_SUITE_END()