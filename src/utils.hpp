#pragma once

#include <iostream>
#include <vector>
#include <chrono>

class Timer {
private:
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::time_point<clock_t> begin;

public:
    Timer() : begin(clock_t::now()) {
    }

    void reset() {
        begin = clock_t::now();
    }

    [[nodiscard]] double elapsed() const {
        return std::chrono::duration_cast<second_t>(clock_t::now() - begin).count();
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &s) {
    os << "[";
    bool first = true;
    for (const auto &x : s) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << x;
    }
    return os << "]";
}
