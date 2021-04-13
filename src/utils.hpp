#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <memory>

class Timer {
private:
    using clock_t = std::chrono::high_resolution_clock;

    std::chrono::time_point<clock_t> begin;
    const std::string actionName;

public:
    Timer(std::string actionName) : begin(clock_t::now()), actionName(std::move(actionName)) {
    }

    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double, std::micro>(clock_t::now() - begin).count() / 1e6;
    }

    ~Timer() {
        std::cout.precision(10);
        std::cout << actionName << " took " << std::fixed << elapsed() << " seconds." << std::endl;
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

u_int32_t
intersection(const std::vector<u_int32_t> &predictions, const std::vector<u_int32_t> &ground_truth);

template<typename ... Args>
std::string string_format(const std::string &format, Args ... args) {
    int size_s = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>(size);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}