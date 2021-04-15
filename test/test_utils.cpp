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
    BOOST_TEST(recall > 0.5);
    // returns query time in mulliseconds
    return {1000 * query_time, recall};
}

void profileSift(const std::string &siftDir, const std::string &saveDir) {
    std::cout << "Profiling ANN on SIFT1M" << std::endl;

    const size_t dimension = 128;
    const size_t epochs = 100;
    const float tol = 1e-5;

    const size_t baseVectorCount = 1e6;
    const size_t queryVectorCount = 1e4;
    const size_t learnVectorCount = 1e4;

    const size_t answerDimension = 100;

    const std::vector<size_t> clusterCounts = {100, 1000};
    const std::vector<size_t> nearestVectorsCounts = {1, 100};
    const std::vector<float> clusterCountParts = {0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25};

    for (size_t clusterCount: clusterCounts) {
        for (size_t nearestVectorsCount: nearestVectorsCounts) {
            PaseIVFFlat<float> tempPase(dimension, clusterCount);
            double buildTime{};

            Parser<float> learnDataParser(siftDir + "sift_learn.fvecs", dimension,
                                          learnVectorCount);
            const std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();

            Parser<float> baseDataParser(siftDir + "sift_base.fvecs", dimension, baseVectorCount);
            const std::vector<std::vector<float>> baseData = baseDataParser.parse();

            Parser<float> testParser(siftDir + "sift_query.fvecs", dimension, queryVectorCount);
            const std::vector<std::vector<float>> testData = testParser.parse();

            Parser<u_int32_t> answerParser(siftDir + "sift_groundtruth.ivecs", answerDimension,
                                           queryVectorCount);
            const std::vector<std::vector<u_int32_t>> parsedTestAnswers = answerParser.parse();

            std::vector<u_int32_t> ids(baseData.size());
            std::iota(ids.begin(), ids.end(), 0);

            {
                Timer t("Build");
                tempPase.buildIndex(dataToLearn, baseData, ids, epochs, tol);
                buildTime = t.elapsed();
            }

            std::ofstream file;
            std::string filename = string_format(saveDir + "sift_cc%d_k%d.csv", clusterCount,
                                                 nearestVectorsCount);
            file.open(filename, std::ofstream::out | std::ofstream::app);
            if (file.is_open()) {
                file << ",Selected clusters,\"Build time, s\",\"Query time, ms\",Recall\n";
            } else {
                throw std::runtime_error("cannot open file: " + filename);
            }
            size_t index = 0;
            for (float clusterCountPart: clusterCountParts) {
                auto clusterCountToSelect = static_cast<size_t>(clusterCountPart *
                                                                static_cast<float>(clusterCount));

                auto[queryTime, recall] = testSearch(tempPase, queryVectorCount, nearestVectorsCount,
                                                     clusterCountToSelect, testData, parsedTestAnswers);


                auto clusterPercentage = static_cast<size_t>(100.0 * clusterCountPart);
                if (file.is_open()) {
                    file.precision(4);
                    file << index++ << ',' << clusterPercentage << ',' << buildTime << ',' << queryTime << ','
                         << recall << '\n';
                }
            }
            file.close();
        }
    }
}

void profileGist(const std::string &gistDir, const std::string &saveDir) {
    std::cout << "Profiling ANN on GIST1M" << std::endl;

    const size_t dimension = 960;
    const size_t epochs = 100;
    const float tol = 1e-5;

    const size_t baseVectorCount = 1e6;
    const size_t queryVectorCount = 1e4;
    const size_t learnVectorCount = 1e4;

    const size_t answerDimension = 100;

    const std::vector<size_t> clusterCounts = {100, 1000};
    const std::vector<size_t> nearestVectorsCounts = {1, 100};
    const std::vector<float> clusterCountParts = {0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.25};

    for (size_t clusterCount: clusterCounts) {
        for (size_t nearestVectorsCount: nearestVectorsCounts) {
            PaseIVFFlat<float> tempPase(dimension, clusterCount);
            double buildTime{};

            Parser<float> learnDataParser(gistDir + "gist_learn.fvecs", dimension,
                                          learnVectorCount);
            const std::vector<std::vector<float>> dataToLearn = learnDataParser.parse();

            Parser<float> baseDataParser(gistDir + "gist_base.fvecs", dimension, baseVectorCount);
            const std::vector<std::vector<float>> baseData = baseDataParser.parse();

            Parser<float> testParser(gistDir + "gist_query.fvecs", dimension, queryVectorCount);
            const std::vector<std::vector<float>> testData = testParser.parse();

            Parser<u_int32_t> answerParser(gistDir + "gist_groundtruth.ivecs", answerDimension,
                                           queryVectorCount);
            const std::vector<std::vector<u_int32_t>> parsedTestAnswers = answerParser.parse();

            std::vector<u_int32_t> ids(baseData.size());
            std::iota(ids.begin(), ids.end(), 0);

            {
                Timer t("Build");
                tempPase.buildIndex(dataToLearn, baseData, ids, epochs, tol);
                buildTime = t.elapsed();
            }

            std::ofstream file;
            std::string filename = string_format(saveDir + "gist_cc%d_k%d.csv", clusterCount,
                                                 nearestVectorsCount);
            file.open(filename, std::ofstream::out | std::ofstream::app);
            if (file.is_open()) {
                file << ",Selected clusters,\"Build time, s\",\"Query time, ms\",Recall\n";
            } else {
                throw std::runtime_error("cannot open file: " + filename);
            }
            size_t index = 0;
            for (float clusterCountPart: clusterCountParts) {
                auto clusterCountToSelect = static_cast<size_t>(clusterCountPart *
                                                                static_cast<float>(clusterCount));

                auto[queryTime, recall] = testSearch(tempPase, queryVectorCount, nearestVectorsCount,
                                                     clusterCountToSelect, testData, parsedTestAnswers);


                auto clusterPercentage = static_cast<size_t>(100.0 * clusterCountPart);
                if (file.is_open()) {
                    file.precision(4);
                    file << index++ << ',' << clusterPercentage << ',' << buildTime << ',' << queryTime << ','
                         << recall << '\n';
                }
            }
            file.close();
        }
    }
}

