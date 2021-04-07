#include <string>
#include <fstream>


template<typename T>
class Parser {
    std::string pathToFile;
    size_t dimension;
    size_t vectorCount;

public:
    explicit Parser(std::string pathToFile, size_t dimension, size_t vectorCount)
            : pathToFile(std::move(pathToFile)), dimension(dimension), vectorCount(vectorCount) {}

    std::vector<std::vector<T>> parse() {
        std::vector<std::vector<T>> result(vectorCount);
        std::ifstream data(pathToFile, std::ios::binary);
        if (!data.is_open()) {
            throw std::runtime_error("Failed to open file '" + pathToFile + "'.");
        }

        char buf[sizeof(T) * dimension + 4];
        for (int i = 0; i < vectorCount; ++i) {
            result[i] = std::vector<T>(dimension);
            data.read(buf, sizeof(buf));
            for (int j = 0; j < dimension; ++j) {
                std::memcpy(&result[i][j], buf + 4 + j * sizeof(T), sizeof(T));
            }
        }
        return result;
    }
};
