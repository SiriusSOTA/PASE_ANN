#include "thread_pool.h"
#include <boost/asio/post.hpp>


using namespace boost;

ThreadPool::ThreadPool()
        : work_guard_(asio::make_work_guard(io_context_)) {
    auto threadCount = std::thread::hardware_concurrency();
    StartWorkerThreads(threadCount);
}

ThreadPool::~ThreadPool() {
    Join();
}

void ThreadPool::Submit(Task task) {
    asio::post(io_context_, std::move(task));
}

void ThreadPool::Join() {
    if (joined_) {
        return;
    }
    work_guard_.reset();
    for (auto& worker : workers_) {
        worker.join();
    }
    joined_ = true;
}

void ThreadPool::Work() {
    io_context_.run();  // Invoke posted handlers
}

void ThreadPool::StartWorkerThreads(size_t count) {
    for (size_t i = 0; i < count; ++i) {
        workers_.emplace_back([this]() {
            Work();
        });
    }
}
