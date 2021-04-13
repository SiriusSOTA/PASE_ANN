#include "parser.hpp"
#include "pase.hpp"
#include "page.hpp"
#include "utils.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <numeric>


BOOST_AUTO_TEST_SUITE(VectorSearch)

    BOOST_AUTO_TEST_CASE(TestSearch) {
        const size_t centroidTuplesPerPage = 204;
        const size_t dimension = 128;
        const size_t clusterCount = 100;
        const size_t epochs = 100;
        const size_t nearestVectorsCount = 100;
        const size_t baseVectorCount = 1e4;
        const size_t queryVectorCount = 100;
        const size_t learnVectorCount = 25e3;
        const size_t clusterCountToSelect = 10;
        const float tol = 1e-4;

        BOOST_TEST(Page<CentroidTuple<float>>::calcTuplesSize() == centroidTuplesPerPage);

        PaseIVFFlat<float> pase(dimension, clusterCount);

        Parser<float> learnDataParser("../../test/test_data/siftsmall_learn.fvecs", dimension, learnVectorCount);
        std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();

        Parser<float> baseDataParser("../../test/test_data/siftsmall_base.fvecs", dimension, baseVectorCount);
        std::vector<std::vector<float>> baseData = baseDataParser.parse();

        Parser<float> testParser("../../test/test_data/siftsmall_query.fvecs", dimension, queryVectorCount);
        std::vector<std::vector<float>> testData = testParser.parse();

        Parser<int> answerParser("../../test/test_data/siftsmall_groundtruth.ivecs", nearestVectorsCount,
                                 queryVectorCount);
        std::vector<std::vector<int>> parsedTestAnswers = answerParser.parse();

        std::vector<u_int32_t> ids(baseData.size());
        std::iota(ids.begin(), ids.end(), 0);
        pase.buildIndex(dataToLearn, baseData, ids, epochs, tol);

        Timer t;
        std::vector<size_t> matchCounter(queryVectorCount, 0);
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
        size_t time = t.elapsed();
        std::cout << "Search done in " << time << " seconds" << std::endl;
        std::cout << "Average running time: " << static_cast<float>(time) / queryVectorCount << std::endl;

        // the rate of the true item in top 100 search results
        auto matched = std::accumulate(matchCounter.begin(), matchCounter.end(), 0.);
        auto recall = matched / queryVectorCount / nearestVectorsCount;
        std::cout << "Recall metric R1@100: " << recall << std::endl;
    }

BOOST_AUTO_TEST_SUITE_END()
