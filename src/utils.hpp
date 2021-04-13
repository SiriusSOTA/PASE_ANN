#pragma once

#include <iostream>
#include <vector>
#include <chrono>

class Timer {
private:
    using clock_t = std::chrono::high_resolution_clock;

    std::chrono::time_point<clock_t> begin;
    const std::string actionName;

public:
    Timer(std::string actionName) : begin(clock_t::now()), actionName(std::move(actionName)) {
    }

    ~Timer() {
        std::cout.precision(10);
        std::cout << actionName << " took " << std::fixed << elapsed() << " seconds." << std::endl;
    }

private:
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double, std::micro>(clock_t::now() - begin).count() / 1e6;
    }

//    void reset() {
//        begin = clock_t::now();
//    }
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
