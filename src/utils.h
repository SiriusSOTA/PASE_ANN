#ifndef PASE_ANN_UTILS_H
#define PASE_ANN_UTILS_H

#include <iostream>
#include <vector>

template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& s) {
    os << "[";
    bool first = true;
    for (const auto& x : s) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << x;
    }
    return os << "]";
}

#endif //PASE_ANN_UTILS_H
