#include "parser.hpp"
#include "pase.hpp"
#include "utils.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <numeric>


void testSearch(const PaseIVFFlat<float> &pase,
                const size_t queryVectorCount,
                const size_t nearestVectorsCount,
                const size_t clusterCountToSelect,
                const std::vector<std::vector<float>> &testData,
                const std::vector<std::vector<int>> &parsedTestAnswers) {
    std::cout << "\nLaunch search on " << queryVectorCount << " query vectors.\nCluster count to scan: "
              << clusterCountToSelect
              << ".\nLook for " << nearestVectorsCount << " neighbours." << std::endl;
    std::vector<size_t> matchCounter(queryVectorCount, 0);
    {
        Timer t("Search");
        for (size_t i = 0; i < queryVectorCount; ++i) {
            std::vector<u_int32_t> searchVectors = pase.findNearestVectorIds(testData[i], nearestVectorsCount,
                                                                             clusterCountToSelect);
            std::unordered_set<size_t> answers;
            for (size_t j = 0; j < nearestVectorsCount; ++j) {
                answers.insert(searchVectors[j]);
            }
            for (size_t j = 0; j < nearestVectorsCount; ++j) {
                if (answers.find(parsedTestAnswers[i][j]) != answers.end()) {
                    ++matchCounter[i];
                }
            }
        }
    }
    auto matched = std::accumulate(matchCounter.begin(), matchCounter.end(), 0.);
    auto recall = matched / queryVectorCount / nearestVectorsCount;
    // the rate of the true item in top nearestVectorsCount search results
    std::cout << "Recall metric R1@" << nearestVectorsCount << ": " << recall << std::endl;
    BOOST_TEST(recall > 0.8);
}


BOOST_AUTO_TEST_SUITE(VectorSearch)

    BOOST_AUTO_TEST_CASE(ANN_SIFT10K) {
        std::cout << "Test ANN_SIFT10K" << std::endl;

        const size_t dimension = 128;
        const size_t clusterCount = 100;
        const size_t epochs = 100;
        const size_t nearestVectorsCount = 100;

        const size_t baseVectorCount = 1e4;
        const size_t queryVectorCount = 100;
        const size_t learnVectorCount = 25e3;

        const size_t clusterCountToSelect = 10;
        const float tol = 1e-4;

        PaseIVFFlat<float> pase(dimension, clusterCount);

        Parser<float> learnDataParser("../../test/test_data/sift_small/siftsmall_learn.fvecs", dimension, learnVectorCount);
        const std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();

        Parser<float> baseDataParser("../../test/test_data/sift_small/siftsmall_base.fvecs", dimension, baseVectorCount);
        const std::vector<std::vector<float>> baseData = baseDataParser.parse();

        Parser<float> testParser("../../test/test_data/sift_small/siftsmall_query.fvecs", dimension, queryVectorCount);
        const std::vector<std::vector<float>> testData = testParser.parse();

        Parser<int> answerParser("../../test/test_data/sift_small/siftsmall_groundtruth.ivecs", nearestVectorsCount,
                                 queryVectorCount);
        const std::vector<std::vector<int>> parsedTestAnswers = answerParser.parse();

        std::vector<u_int32_t> ids(baseData.size());
        std::iota(ids.begin(), ids.end(), 0);
        pase.buildIndex(dataToLearn, baseData, ids, epochs, tol);

        testSearch(pase, queryVectorCount, nearestVectorsCount, clusterCountToSelect, testData, parsedTestAnswers);
        testSearch(pase, queryVectorCount, 1, clusterCountToSelect, testData, parsedTestAnswers);
    }

//    BOOST_AUTO_TEST_CASE(ANN_SIFT1M) {
//        std::cout << "Test ANN_SIFT1M" << std::endl;
//
//        const size_t dimension = 128;
//        const size_t clusterCount = 1000;
//        const size_t epochs = 100;
//        const size_t nearestVectorsCount = 100;
//
//        const size_t baseVectorCount = 1e6;
//        const size_t queryVectorCount = 1e3;
//        const size_t learnVectorCount = 1e4;
//
//        const size_t clusterCountToSelect = 10;
//        const float tol = 1e-4;
//
//        PaseIVFFlat<float> pase(dimension, clusterCount);
//
//        Parser<float> learnDataParser("../../test/test_data/sift/sift_learn.fvecs", dimension, learnVectorCount);
//        const std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();
//
//        Parser<float> baseDataParser("../../test/test_data/sift/sift_base.fvecs", dimension, baseVectorCount);
//        const std::vector<std::vector<float>> baseData = baseDataParser.parse();
//
//        Parser<float> testParser("../../test/test_data/sift/sift_query.fvecs", dimension, queryVectorCount);
//        const std::vector<std::vector<float>> testData = testParser.parse();
//
//        Parser<int> answerParser("../../test/test_data/sift/sift_groundtruth.ivecs", nearestVectorsCount,
//                                 queryVectorCount);
//        const std::vector<std::vector<int>> parsedTestAnswers = answerParser.parse();
//
//        std::vector<u_int32_t> ids(baseData.size());
//        std::iota(ids.begin(), ids.end(), 0);
//        pase.buildIndex(dataToLearn, baseData, ids, epochs, tol);
//
//        testSearch(pase, queryVectorCount, nearestVectorsCount, clusterCountToSelect, testData, parsedTestAnswers);
//        testSearch(pase, queryVectorCount, 1, clusterCountToSelect, testData, parsedTestAnswers);
//    }

BOOST_AUTO_TEST_SUITE_END()
