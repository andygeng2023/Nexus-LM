#pragma once
#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include <filesystem>
#include <cstdint>
namespace nexus {
struct TrainConfig {
    int64_t steps = 1000;
    int64_t batch_size = 8;
    int64_t sequence_length = 128;
    int64_t log_every = 25;
    int64_t save_every = 250;
    int64_t validation_batches = 8;
    double learning_rate = 3e-4;
    uint64_t seed = 42;
};
class Trainer {
public:
    Trainer(NexusModel model, TrainConfig config, torch::Device device);
    void train_text_file(const std::filesystem::path& corpus,
                         const std::filesystem::path& checkpoint,
                         const BpeTokenizer& tokenizer);
private:
    NexusModel model_;
    TrainConfig config_;
    torch::Device device_;
};
}
