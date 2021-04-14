#include "test_utils.hpp"

std::pair<double, float>
testSearch(const PaseIVFFlat<float> &pase, const size_t queryVectorCount, const size_t nearestVectorsCount,
           const size_t clusterCountToSelect, const std::vector<std::vector<float>> &testData,
           const std::vector<std::vector<u_int32_t>> &parsedTestAnswers) {
    std::cout << "\nLaunch search on " << queryVectorCount << " query vectors.\nCluster count to scan: "
              << clusterCountToSelect
              << ".\nLook for " << nearestVectorsCount << " neighbours." << std::endl;
    std::vector<size_t> matchCounter(queryVectorCount, 0);
    double query_time{};
    {
        Timer t("Search");
        for (size_t i = 0; i < queryVectorCount; ++i) {
            std::vector<u_int32_t> searchVectors = pase.findNearestVectorIds(testData[i], nearestVectorsCount,
                                                                             clusterCountToSelect);
            matchCounter[i] += intersection(searchVectors, parsedTestAnswers[i]);
        }
        query_time = t.elapsed();
        query_time /= static_cast<double>(queryVectorCount);
    }
    double matched = std::accumulate(matchCounter.begin(), matchCounter.end(), 0.);
    auto recall = matched / static_cast<double>(queryVectorCount) / static_cast<double>(nearestVectorsCount);
    // the rate of the true item in top nearestVectorsCount search results
    std::cout << "R1@" << nearestVectorsCount << ": " << recall << std::endl;
    BOOST_TEST(recall > 0.7);
    return {query_time, recall};
}

