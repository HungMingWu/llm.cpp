module;
#include <chrono>

export module ggml:stopwatch;

export
{
    class Stopwatch {
    public:
        Stopwatch() = default;

        template <typename duration = std::chrono::microseconds>
        int64_t get_elapsed() const {
            auto end_time_ = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<duration>(end_time_ - start_time_).count();
        }

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time_ = std::chrono::high_resolution_clock::now();
    };
}