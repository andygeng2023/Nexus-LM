#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include "nexus/trainer.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}

std::string read_tokenizer_sample(const std::string& path, std::size_t max_bytes) {
    if (max_bytes == 0) return read_text(path);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    const auto file_size = static_cast<std::size_t>(in.tellg());
    if (file_size <= max_bytes) {
        in.seekg(0);
        return std::string((std::istreambuf_iterator<char>(in)), {});
    }
    constexpr std::size_t chunks = 16;
    const auto chunk_size = std::max<std::size_t>(1, max_bytes / chunks);
    std::string sample;
    sample.reserve(chunk_size * chunks + chunks);
    for (std::size_t i = 0; i < chunks; ++i) {
        const auto start = (file_size - chunk_size) * i / (chunks - 1);
        in.clear();
        in.seekg(static_cast<std::streamoff>(start));
        std::string part(chunk_size, '\0');
        in.read(part.data(), static_cast<std::streamsize>(chunk_size));
        part.resize(static_cast<std::size_t>(in.gcount()));
        if (!sample.empty()) sample.push_back('\n');
        sample += part;
    }
    return sample;
}

std::vector<int64_t> encode_file(const std::string& path, const nexus::BpeTokenizer& tokenizer) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    std::vector<int64_t> ids;
    std::string line;
    while (std::getline(in, line)) {
        if (!in.eof()) line.push_back('\n');
        auto encoded = tokenizer.encode(line);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
    }
    return ids;
}

void usage() {
    std::cout << "NEXUS-LM 0.3.0\n"
              << "  nexus tokenizer-train <corpus.txt> <tokenizer.bpe> [vocab-size=1024] [sample-bytes=200000]\n"
              << "  nexus train <corpus.txt> <checkpoint.pt> <tokenizer.bpe> [steps=1000] [sequence=128] [batch=8] [validation.txt]\n"
              << "  nexus evaluate <checkpoint.pt> <tokenizer.bpe> <test.txt> [max-batches=256]\n"
              << "  nexus generate <checkpoint.pt> <tokenizer.bpe> <prompt> [new-tokens=100]\n";
}

nexus::ModelConfig default_model_config(const nexus::BpeTokenizer& tokenizer,
                                        int64_t context = 256) {
    nexus::ModelConfig config;
    config.vocab_size = tokenizer.vocab_size();
    config.context = std::max<int64_t>(256, context);
    return config;
}

int evaluate(const std::string& checkpoint, const std::string& tokenizer_path,
             const std::string& test_path, int64_t max_batches) {
    nexus::BpeTokenizer tokenizer;
    tokenizer.load(tokenizer_path);
    auto config = default_model_config(tokenizer);
    auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                               : torch::Device(torch::kCPU);
    nexus::NexusModel model(config);
    torch::load(model, checkpoint);
    model->to(device);
    model->eval();

    const auto ids = encode_file(test_path, tokenizer);
    if (ids.size() <= static_cast<std::size_t>(config.context + 1))
        throw std::runtime_error("test split is too short for evaluation context");
    const auto total_windows = static_cast<int64_t>(
        (ids.size() - 1 + static_cast<std::size_t>(config.context) - 1) /
        static_cast<std::size_t>(config.context));
    const auto windows = max_batches > 0 ? std::min(total_windows, max_batches) : total_windows;
    double weighted_loss = 0.0;
    int64_t evaluated_tokens = 0;
    torch::NoGradGuard guard;
    for (int64_t i = 0; i < windows; ++i) {
        const auto start = static_cast<std::size_t>(i * config.context);
        const auto length = std::min<std::size_t>(
            static_cast<std::size_t>(config.context), ids.size() - start - 1);
        if (length == 0) break;
        std::vector<int64_t> input(ids.begin() + static_cast<std::ptrdiff_t>(start),
                                   ids.begin() + static_cast<std::ptrdiff_t>(start + length));
        std::vector<int64_t> target(ids.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                    ids.begin() + static_cast<std::ptrdiff_t>(start + length + 1));
        auto x = torch::tensor(input, torch::kInt64).unsqueeze(0).to(device);
        auto y = torch::tensor(target, torch::kInt64).unsqueeze(0).to(device);
        auto logits = model->forward(x);
        const auto loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, config.vocab_size}), y.reshape({-1})).item<double>();
        weighted_loss += loss * static_cast<double>(length);
        evaluated_tokens += static_cast<int64_t>(length);
    }
    const auto mean_loss = weighted_loss / static_cast<double>(evaluated_tokens);
    std::cout << "test_tokens=" << evaluated_tokens
              << " test_loss=" << mean_loss
              << " perplexity=" << std::exp(mean_loss) << '\n';
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 0; }
        const std::string command = argv[1];
        if (command == "tokenizer-train") {
            if (argc < 4) { usage(); return 2; }
            const auto target = argc > 4 ? std::stoll(argv[4]) : 1024;
            const auto sample_bytes = argc > 5 ? std::stoull(argv[5]) : 200000;
            const auto sample = read_tokenizer_sample(argv[2], sample_bytes);
            if (sample.size() < 2) throw std::runtime_error("corpus is too short to train a tokenizer");
            nexus::BpeTokenizer tokenizer;
            tokenizer.train(sample, target);
            tokenizer.save(argv[3]);
            std::cout << "BPE vocabulary=" << tokenizer.vocab_size()
                      << " merges=" << tokenizer.merge_count()
                      << " sample_bytes=" << sample.size()
                      << " saved=" << argv[3] << '\n';
            return 0;
        }
        if (command == "train") {
            if (argc < 5) { usage(); return 2; }
            nexus::BpeTokenizer tokenizer;
            tokenizer.load(argv[4]);
            nexus::TrainConfig training;
            if (argc > 5) training.steps = std::stoll(argv[5]);
            if (argc > 6) training.sequence_length = std::stoll(argv[6]);
            if (argc > 7) training.batch_size = std::stoll(argv[7]);
            nexus::ModelConfig model_config = default_model_config(tokenizer, training.sequence_length);
            auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                       : torch::Device(torch::kCPU);
            torch::manual_seed(static_cast<int64_t>(training.seed));
            nexus::NexusModel model(model_config);
            const std::filesystem::path validation = argc > 8 ? argv[8] : "";
            nexus::Trainer(model, training, device).train_text_file(
                argv[2], argv[3], tokenizer, validation);
            return 0;
        }
        if (command == "evaluate") {
            if (argc < 5) { usage(); return 2; }
            const auto batches = argc > 5 ? std::stoll(argv[5]) : 256;
            return evaluate(argv[2], argv[3], argv[4], batches);
        }
        if (command == "generate") {
            if (argc < 5) { usage(); return 2; }
            nexus::BpeTokenizer tokenizer;
            tokenizer.load(argv[3]);
            auto model_config = default_model_config(tokenizer);
            auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                       : torch::Device(torch::kCPU);
            nexus::NexusModel model(model_config);
            torch::load(model, argv[2]);
            model->to(device);
            model->eval();
            const auto ids = tokenizer.encode(argv[4]);
            if (ids.empty()) throw std::runtime_error("prompt must not be empty");
            auto input = torch::tensor(ids, torch::kInt64).unsqueeze(0).to(device);
            const auto count = argc > 5 ? std::stoll(argv[5]) : 100;
            auto generated = model->generate(input, count).to(torch::kCPU).squeeze(0).contiguous();
            std::vector<int64_t> output(generated.data_ptr<int64_t>(),
                                        generated.data_ptr<int64_t>() + generated.numel());
            std::cout << tokenizer.decode(output) << '\n';
            return 0;
        }
        usage();
        std::cerr << "unknown command: " << command << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "NEXUS error: " << error.what() << '\n';
        return 1;
    }
}
