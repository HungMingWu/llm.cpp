module;
#include <chrono>

export module ggml:stopwatch;

export
{
    class Stopwatch {
    public:
        void start() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            end_time_ = std::chrono::high_resolution_clock::now();
        }

        template <typename duration = std::chrono::microseconds>
        int64_t get_elapsed() const {
            return std::chrono::duration_cast<duration>(end_time_ - start_time_).count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    };
}