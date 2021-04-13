//
// Created by dmitry on 13.04.2021.
//

#include "utils.hpp"

u_int32_t
intersection(const std::vector<u_int32_t> &predictions, const std::vector<u_int32_t> &ground_truth) {
    u_int32_t intersection = 0;
    size_t top_k = predictions.size();
    std::unordered_set<u_int32_t> preds(predictions.begin(), predictions.end());
    for (size_t i = 0; i < top_k; ++i) {
        if (preds.find(ground_truth[i]) != preds.end()) {
            ++intersection;
        }
    }
    return intersection;
}

