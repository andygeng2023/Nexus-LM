#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include <torch/torch.h>
#include <cassert>
#include <filesystem>
#include <iostream>

int main() {
    try {
        const std::string corpus = "the quick brown fox jumps over the lazy dog. ";
        nexus::ByteTokenizer bytes;
        assert(bytes.decode(bytes.encode(corpus)) == corpus);

        nexus::BpeTokenizer bpe;
        std::string repeated;
        for (int i = 0; i < 30; ++i) repeated += corpus;
        bpe.train(repeated, 300);
        assert(bpe.vocab_size() > 256);
        assert(bpe.decode(bpe.encode(repeated)) == repeated);
        const auto path = std::filesystem::temp_directory_path() / "nexus-test-tokenizer.bpe";
        bpe.save(path);
        nexus::BpeTokenizer loaded; loaded.load(path);
        assert(loaded.vocab_size() == bpe.vocab_size());
        assert(loaded.decode(loaded.encode(repeated)) == repeated);
        std::filesystem::remove(path);

        torch::manual_seed(7);
        nexus::SelectiveStateSpace ssm(16);
        auto sequence = torch::randn({1, 8, 16}, torch::TensorOptions().requires_grad(true));
        auto altered = sequence.detach().clone(); altered.slice(1, 5, 8).add_(2.0);
        auto y1 = ssm->forward(sequence);
        auto y2 = ssm->forward(altered);
        assert(torch::allclose(y1.slice(1, 0, 5), y2.slice(1, 0, 5), 1e-5, 1e-6));
        y1.square().mean().backward();
        assert(ssm->named_parameters(true).size() > 0);

        nexus::ModelConfig cfg; cfg.vocab_size = loaded.vocab_size(); cfg.width = 32;
        cfg.layers = 2; cfg.heads = 4; cfg.context = 16; cfg.experts = 2; cfg.top_k = 1;
        nexus::NexusModel model(cfg);
        auto tokens = torch::randint(0, cfg.vocab_size, {2, 12}, torch::TensorOptions().dtype(torch::kInt64));
        auto logits = model->forward(tokens);
        assert(logits.sizes() == torch::IntArrayRef({2, 12, cfg.vocab_size}));
        auto future_changed = tokens.clone(); future_changed.index_put_({0, 8}, (tokens.index({0, 8}) + 1) % cfg.vocab_size);
        auto eval_logits_1 = model->forward(tokens);
        auto eval_logits_2 = model->forward(future_changed);
        assert(torch::allclose(eval_logits_1.index({0, torch::indexing::Slice(0, 8)}),
                               eval_logits_2.index({0, torch::indexing::Slice(0, 8)}), 1e-5, 1e-6));
        auto loss = torch::nn::functional::cross_entropy(logits.reshape({-1, cfg.vocab_size}), tokens.reshape({-1}));
        loss.backward(); assert(loss.item<double>() > 0.0);

        nexus::TransformerBaseline baseline(cfg, 3);
        auto reference_logits = baseline->forward(tokens);
        assert(reference_logits.sizes() == logits.sizes());
        std::cout << "NEXUS tests passed; BPE vocab=" << loaded.vocab_size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n'; return 1;
    }
}
