#include "clustering.hpp"
#include "parser.hpp"
#include "pase.hpp"
#include "page.hpp"
#include "utils.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

BOOST_AUTO_TEST_SUITE(VectorSearch)

BOOST_AUTO_TEST_CASE(TestSearch) {
    const size_t centroidTuplesPerPage = 204;
    const size_t dimension = 128;
    const size_t clusterCount = 100;
    const size_t epochs = 100;
    const size_t nearestVectorsCount = 100;
    const size_t vectorCount = 1e4;
    const size_t clusterCountToSelect = 1e3;
    const float tol = 1e-4;

    BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

    PaseIVFFlat<float> pase(dimension, clusterCount);
    Parser<float> dataParser("../../test/test_data/siftsmall_base.fvecs", dimension, vectorCount);
    std::vector<const std::vector<float>> dataParsed = dataParser.parse();
    IVFFlatClusterData<float> clusterData = kMeans<float>(dataParsed, clusterCount, epochs, tol);

    Parser<float> testParser("../../test/test_data/siftsmall_query.fvecs", dimension, vectorCount);
    std::vector<const std::vector<float>> parsedTestData = testParser.parse();

    Parser<u_int32_t> answerParser("../../test/test_data/siftsmall_groundtruth.ivecs", nearestVectorsCount, vectorCount);
    std::vector<const std::vector<u_int32_t>> parsedTestAnswers = answerParser.parse();
    
    Timer t;
    for (size_t i = 0; i < vectorCount; ++i) {
        std::vector<u_int32_t>& searchVectors = clusterData.findNearestVectorIds(dataParsed[i], nearestVectorsCount, clusterCountToSelect);
        for (size_t j = 0; j < nearestVectorsCount; ++j){
            BOOST_TEST(searchVectors[j] == parsedTestAnswers[i][j]);
        }
    }
    std::cout << "IVFFlat done in " << t.elapsed() << " seconds" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
