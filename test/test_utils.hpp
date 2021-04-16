#pragma once

#include <vector>
#include "pase.hpp"

std::pair<double, float> testSearch(const PaseIVFFlat<float> &pase,
                                    const size_t queryVectorCount,
                                    const size_t nearestVectorsCount,
                                    const size_t clusterCountToSelect,
                                    const std::vector<std::vector<float>> &testData,
                                    const std::vector<std::vector<u_int32_t>> &parsedTestAnswers);


void profileSift(const std::string& siftDir, const std::string& saveDir);

void profileGist(const std::string& gistDir, const std::string& saveDir);

