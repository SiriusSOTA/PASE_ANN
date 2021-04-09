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
    static size_t calcVectorCount(size_t dimension) {
        return calcTuplesSize() * sizeof(T) / (sizeof(T) * dimension + 4);
    }
    inline bool hasNextPage() {
        return nextPage != nullptr;
    }
    inline typename std::vector<T>::iterator getEndTuples(size_t dimension) {
        return tuples.begin() + calcVectorCount(dimension) * dimension;
    }
};

template<typename T>
using DataPage = Page<T>;

template<typename T>
struct CentroidTuple {
    std::vector<T> vec;
    DataPage<T>* firstDataPage;
    size_t vectorCount = 0;

    ~CentroidTuple() {
        auto curDataPage = firstDataPage;
        if (curDataPage == nullptr) {
            return;
        }
        DataPage<T> *nextDataPage = nullptr;
        while (curDataPage->hasNextPage()) {
            nextDataPage = curDataPage->nextPage;
            delete curDataPage;
            curDataPage = nextDataPage;
        }
        delete curDataPage;
    }
};

template<typename T>
using CentroidPage = Page<CentroidTuple<T>>;
