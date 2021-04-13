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
#include <map>

BOOST_AUTO_TEST_SUITE(VectorSearch)

BOOST_AUTO_TEST_CASE(TestSearch) {
    const size_t centroidTuplesPerPage = 204;
    const size_t dimension = 128;
    const size_t clusterCount = 100;
    const size_t epochs = 100;
    const size_t nearestVectorsCount = 100;
    const size_t vectorCount = 1e4;
    const size_t clusterCountToSelect = 10;
    const float tol = 1e-4;

    BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

    PaseIVFFlat<float> pase(dimension, clusterCount);
    Parser<float> learnDataParser("../../test/test_data/siftsmall_learn.fvecs", dimension, vectorCount);
    std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();
    Parser<float> baseDataParser("../../test/test_data/siftsmall_base.fvecs", dimension, vectorCount);
    std::vector<std::vector<float>> baseData = baseDataParser.parse();
    std::vector<u_int32_t> ids(baseData.size());
    std::iota(ids.begin(), ids.end(), 0);
    pase.buildIndex(dataToLearn, baseData, ids, epochs, tol);

    Parser<float> testParser("../../test/test_data/siftsmall_query.fvecs", dimension, vectorCount);
    std::vector<std::vector<float>> testData = testParser.parse();

    Parser<int> answerParser("../../test/test_data/siftsmall_groundtruth.ivecs", nearestVectorsCount, vectorCount);
    std::vector<std::vector<int>> parsedTestAnswers = answerParser.parse();
    Timer t;
    std::vector<size_t> matchCounter(vectorCount, 0);
    for (size_t i = 0; i < vectorCount; ++i) {
        std::vector<u_int32_t> searchVectors = pase.findNearestVectorIds(testData[i], nearestVectorsCount, clusterCountToSelect);
        std::map<size_t, bool> answer;
        for (size_t j = 0; j < nearestVectorsCount; ++j){
            answer[searchVectors[j]] = 1;
        }
        for (size_t j = 0; j < nearestVectorsCount; ++j) {
            if (answer[parsedTestAnswers[i][j]])
                matchCounter[i]++;
        }
    }
    size_t time = t.elapsed();
    std::cout << "Search done in " << time << " seconds" << std::endl;
    std::cout << "Average running time" << time / vectorCount << std::endl;
    for (size_t i = 0; i < vectorCount; ++i) {
        std::cout << "match: " << matchCounter[i] << " out of 100" << std::endl;
    }
}

BOOST_AUTO_TEST_SUITE_END()
