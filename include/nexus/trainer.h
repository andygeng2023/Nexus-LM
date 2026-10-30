#pragma once
#include "nexus/model.h"
#include <filesystem>
namespace nexus {
struct TrainConfig {int64_t steps=1000,batch_size=8,sequence_length=128,log_every=25,save_every=250;double learning_rate=3e-4;uint64_t seed=42;};
class Trainer {
public: Trainer(NexusModel,TrainConfig,torch::Device); void train_text_file(const std::filesystem::path&,const std::filesystem::path&);
private: NexusModel model_;TrainConfig config_;torch::Device device_;
};
}
