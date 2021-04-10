#pragma once

#include <boost/asio/io_service.hpp>
#include <boost/thread/future.hpp>
#include <vector>
#include <thread>
#include <functional>


using Task = boost::packaged_task<void>;

class ThreadPool {
public:
    ThreadPool();

    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;

    ThreadPool &operator=(const ThreadPool &) = delete;

    void Submit(Task task);

    void Join();

private:
    void StartWorkerThreads(size_t thread_count);

    void Work();

private:
    boost::asio::io_context io_context_;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> work_guard_;
    std::vector<std::thread> workers_;
    bool joined_{false};
};

static ThreadPool& getThreadPool() {
    static ThreadPool threadPool;
    return threadPool;
}
