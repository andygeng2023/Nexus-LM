#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include "nexus/trainer.h"
#include <torch/torch.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}
void usage() {
    std::cout << "NEXUS-LM 0.2.0\n"
              << "  nexus tokenizer-train <corpus.txt> <tokenizer.bpe> [vocab-size=512]\n"
              << "  nexus train <corpus.txt> <checkpoint.pt> <tokenizer.bpe> [steps=1000] [sequence=128] [batch=8]\n"
              << "  nexus generate <checkpoint.pt> <tokenizer.bpe> <prompt> [new-tokens=100]\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) { usage(); return 0; }
        const std::string command = argv[1];
        if (command == "tokenizer-train") {
            if (argc < 4) { usage(); return 2; }
            const auto target = argc > 4 ? std::stoll(argv[4]) : 512;
            const auto raw = read_text(argv[2]);
            if (raw.size() < 2) throw std::runtime_error("corpus is too short to train a tokenizer");
            const auto fit_bytes = std::max<std::size_t>(1, raw.size() * 9 / 10);
            nexus::BpeTokenizer tokenizer;
            tokenizer.train(raw.substr(0, fit_bytes), target);
            tokenizer.save(argv[3]);
            std::cout << "BPE vocabulary=" << tokenizer.vocab_size()
                      << " merges=" << tokenizer.merge_count() << " saved=" << argv[3] << '\n';
            return 0;
        }
        if (command == "train") {
            if (argc < 5) { usage(); return 2; }
            nexus::BpeTokenizer tokenizer; tokenizer.load(argv[4]);
            nexus::TrainConfig training;
            if (argc > 5) training.steps = std::stoll(argv[5]);
            if (argc > 6) training.sequence_length = std::stoll(argv[6]);
            if (argc > 7) training.batch_size = std::stoll(argv[7]);
            nexus::ModelConfig model_config;
            model_config.vocab_size = tokenizer.vocab_size();
            model_config.context = std::max<int64_t>(256, training.sequence_length);
            auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
            torch::manual_seed(static_cast<int64_t>(training.seed));
            nexus::NexusModel model(model_config);
            nexus::Trainer(model, training, device).train_text_file(argv[2], argv[3], tokenizer);
            return 0;
        }
        if (command == "generate") {
            if (argc < 5) { usage(); return 2; }
            nexus::BpeTokenizer tokenizer; tokenizer.load(argv[3]);
            nexus::ModelConfig model_config; model_config.vocab_size = tokenizer.vocab_size();
            auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
            nexus::NexusModel model(model_config);
            torch::load(model, argv[2]); model->to(device); model->eval();
            const auto ids = tokenizer.encode(argv[4]);
            if (ids.empty()) throw std::runtime_error("prompt must not be empty");
            auto input = torch::tensor(ids, torch::kInt64).unsqueeze(0).to(device);
            const auto count = argc > 5 ? std::stoll(argv[5]) : 100;
            auto generated = model->generate(input, count).to(torch::kCPU).squeeze(0).contiguous();
            std::vector<int64_t> output(generated.data_ptr<int64_t>(), generated.data_ptr<int64_t>() + generated.numel());
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
