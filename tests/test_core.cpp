#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include <torch/torch.h>
#include <stdexcept>
#include <filesystem>
#include <iostream>

namespace { void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); } }

int main() {
    try {
        const std::string corpus = "the quick brown fox jumps over the lazy dog. ";
        nexus::ByteTokenizer bytes;
        check(bytes.decode(bytes.encode(corpus)) == corpus, "byte tokenizer round trip");

        nexus::BpeTokenizer bpe;
        std::string repeated;
        for (int i = 0; i < 30; ++i) repeated += corpus;
        bpe.train(repeated, 300);
        check(bpe.vocab_size() > 256, "BPE learned at least one merge");
        check(bpe.decode(bpe.encode(repeated)) == repeated, "BPE in-memory round trip");
        const auto path = std::filesystem::temp_directory_path() / "nexus-test-tokenizer.bpe";
        bpe.save(path);
        nexus::BpeTokenizer loaded; loaded.load(path);
        check(loaded.vocab_size() == bpe.vocab_size(), "BPE vocabulary persisted");
        check(loaded.decode(loaded.encode(repeated)) == repeated, "BPE persisted round trip");
        std::filesystem::remove(path);

        torch::manual_seed(7);
        nexus::SelectiveStateSpace ssm(16);
        auto sequence = torch::randn({1, 8, 16}, torch::TensorOptions().requires_grad(true));
        auto altered = sequence.detach().clone(); altered.slice(1, 5, 8).add_(2.0);
        auto y1 = ssm->forward(sequence);
        auto y2 = ssm->forward(altered);
        check(torch::allclose(y1.slice(1, 0, 5), y2.slice(1, 0, 5), 1e-5, 1e-6), "SSM must be causal");
        y1.square().mean().backward();
        check(ssm->named_parameters(true).size() > 0, "SSM has trainable parameters");

        nexus::ModelConfig cfg; cfg.vocab_size = loaded.vocab_size(); cfg.width = 32;
        cfg.layers = 2; cfg.heads = 4; cfg.context = 16; cfg.experts = 2; cfg.top_k = 1;
        nexus::NexusModel model(cfg);
        auto tokens = torch::randint(0, cfg.vocab_size, {2, 12}, torch::TensorOptions().dtype(torch::kInt64));
        auto logits = model->forward(tokens);
        check(logits.sizes() == torch::IntArrayRef({2, 12, cfg.vocab_size}), "hybrid logits shape");
        auto future_changed = tokens.clone(); future_changed.index_put_({0, 8}, (tokens.index({0, 8}) + 1) % cfg.vocab_size);
        auto eval_logits_1 = model->forward(tokens);
        auto eval_logits_2 = model->forward(future_changed);
        check(torch::allclose(eval_logits_1.index({0, torch::indexing::Slice(0, 8)}),
                              eval_logits_2.index({0, torch::indexing::Slice(0, 8)}), 1e-5, 1e-6),
              "model must not leak future tokens into earlier logits");
        auto loss = torch::nn::functional::cross_entropy(logits.reshape({-1, cfg.vocab_size}), tokens.reshape({-1}));
        loss.backward(); check(loss.item<double>() > 0.0, "hybrid loss is finite and positive");

        nexus::TransformerBaseline baseline(cfg, 3);
        auto reference_logits = baseline->forward(tokens);
        check(reference_logits.sizes() == logits.sizes(), "Transformer baseline logits shape");
        std::cout << "NEXUS tests passed; BPE vocab=" << loaded.vocab_size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n'; return 1;
    }
}
