#include <vector>


static size_t PAGE_SIZE = 8192;  // 8 KB

template<typename T>
struct Page {
    std::vector <T> tuples;
    Page<T> *nextPage = nullptr;

    Page() : tuples(calcTuplesSize()) {}

    inline static size_t calcTuplesSize() {
        return (PAGE_SIZE - sizeof(Page<T>*)) / sizeof(T);
    }
    inline bool hasNextPage() {
        return nextPage != nullptr;
    }
};

template<typename T>
using DataPage = Page<T>;

template<typename T>
struct CentroidTuple {
    std::vector<T> vec;
    DataPage<T>* firstDataPage;
    size_t vectorCount = 0;
};

template<typename T>
using CentroidPage = Page<CentroidTuple<T>>;
