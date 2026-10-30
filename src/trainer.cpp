#include "nexus/trainer.h"
#include <torch/torch.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace nexus {
namespace {
std::vector<int32_t> tokenize_file(const std::filesystem::path& path, const BpeTokenizer& tokenizer) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open corpus: " + path.string());
    std::vector<int32_t> result;
    std::string line;
    while (std::getline(file, line)) {
        // Preserve line boundaries without loading the full multi-hundred-MB
        // corpus into one string or allocating tokenizer work arrays for it.
        if (!file.eof()) line.push_back('\n');
        auto encoded = tokenizer.encode(line);
        result.reserve(result.size() + encoded.size());
        for (const auto id : encoded) result.push_back(static_cast<int32_t>(id));
    }
    return result;
}

torch::Tensor batch_at(const torch::Tensor& data, const std::vector<int64_t>& starts,
                       int64_t length, torch::Device device) {
    std::vector<torch::Tensor> rows;
    rows.reserve(starts.size());
    for (const auto start : starts) rows.push_back(data.slice(0, start, start + length));
    return torch::stack(rows).to(device);
}

double validation_loss(NexusModel& model, const torch::Tensor& data, int64_t length,
                       int64_t batches, uint64_t seed, torch::Device device) {
    if (data.numel() <= length + 1 || batches <= 0) return -1.0;
    model->eval();
    torch::NoGradGuard guard;
    std::mt19937_64 rng(seed);
    const auto max_start = data.size(0) - length - 1;
    std::uniform_int_distribution<int64_t> dist(0, max_start);
    double total = 0.0;
    for (int64_t i = 0; i < batches; ++i) {
        const auto start = dist(rng);
        auto x = data.slice(0, start, start + length).unsqueeze(0).to(device);
        auto y = data.slice(0, start + 1, start + length + 1).unsqueeze(0).to(device);
        auto logits = model->forward(x);
        total += torch::nn::functional::cross_entropy(
            logits.reshape({-1, model->config().vocab_size}), y.reshape({-1})).item<double>();
    }
    model->train();
    return total / static_cast<double>(batches);
}
}

Trainer::Trainer(NexusModel model, TrainConfig config, torch::Device device)
    : model_(std::move(model)), config_(config), device_(device) {
    model_->to(device_);
}

void Trainer::train_text_file(const std::filesystem::path& corpus,
                              const std::filesystem::path& checkpoint,
                              const BpeTokenizer& tokenizer,
                              const std::filesystem::path& validation_corpus) {
    if (config_.steps < 1 || config_.batch_size < 1 || config_.sequence_length < 1 ||
        config_.learning_rate <= 0.0)
        throw std::invalid_argument("invalid training configuration");

    auto train_ids = tokenize_file(corpus, tokenizer);
    std::vector<int32_t> valid_ids;
    if (!validation_corpus.empty()) {
        valid_ids = tokenize_file(validation_corpus, tokenizer);
    } else {
        const std::size_t split = std::max<std::size_t>(1, train_ids.size() * 9 / 10);
        valid_ids.assign(train_ids.begin() + static_cast<std::ptrdiff_t>(split), train_ids.end());
        train_ids.resize(split);
    }

    if (train_ids.size() <= static_cast<std::size_t>(config_.sequence_length + 1))
        throw std::runtime_error("training corpus is too short for the sequence length");
    auto train_data = torch::tensor(train_ids, torch::kInt32);
    auto valid_data = torch::tensor(valid_ids, torch::kInt32);
    const auto train_token_count = train_data.size(0);
    const auto valid_token_count = valid_data.size(0);
    train_ids.clear();
    train_ids.shrink_to_fit();
    valid_ids.clear();
    valid_ids.shrink_to_fit();

    torch::manual_seed(static_cast<int64_t>(config_.seed));
    std::mt19937_64 rng(config_.seed);
    std::uniform_int_distribution<int64_t> start_dist(
        0, train_data.size(0) - config_.sequence_length - 2);
    torch::optim::AdamW optimizer(model_->parameters(),
                                  torch::optim::AdamWOptions(config_.learning_rate));
    model_->train();
    std::cout << "training tokens=" << train_token_count
              << " validation tokens=" << valid_token_count
              << " vocab=" << tokenizer.vocab_size()
              << " device=" << (device_.is_cuda() ? "CUDA" : "CPU") << '\n';

    for (int64_t step = 1; step <= config_.steps; ++step) {
        std::vector<int64_t> starts;
        starts.reserve(static_cast<std::size_t>(config_.batch_size));
        for (int64_t b = 0; b < config_.batch_size; ++b) starts.push_back(start_dist(rng));
        auto x = batch_at(train_data, starts, config_.sequence_length, device_);
        auto target_starts = starts;
        for (auto& start : target_starts) ++start;
        auto y = batch_at(train_data, target_starts, config_.sequence_length, device_);

        optimizer.zero_grad();
        auto logits = model_->forward(x);
        auto loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, model_->config().vocab_size}), y.reshape({-1}));
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        optimizer.step();

        if (step == 1 || (config_.log_every > 0 && step % config_.log_every == 0))
            std::cout << "step " << step << '/' << config_.steps
                      << " train_loss=" << loss.item<double>() << '\n';
        if (config_.save_every > 0 && step % config_.save_every == 0)
            torch::save(model_, checkpoint);
    }
    const auto val = validation_loss(model_, valid_data, config_.sequence_length,
                                     config_.validation_batches, config_.seed + 1, device_);
    if (val >= 0.0) std::cout << "validation_loss=" << val << '\n';
    else std::cout << "validation_loss=unavailable (held-out split too short)\n";
    torch::save(model_, checkpoint);
    std::cout << "checkpoint saved: " << checkpoint << '\n';
}
}
