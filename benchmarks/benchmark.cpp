#include "nexus/model.h"
#include <torch/torch.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
int64_t parameter_count(torch::nn::Module& module, bool experts_only = false) {
    int64_t count = 0;
    for (const auto& item : module.named_parameters(true)) {
        const bool is_expert = item.key().find("experts.") != std::string::npos;
        if (!experts_only || is_expert) count += item.value().numel();
    }
    return count;
}
template <typename Model>
void run_benchmark(const std::string& label, Model& model, const nexus::ModelConfig& cfg,
                   const torch::Tensor& input, const torch::Tensor& target,
                   int64_t warmup, int64_t iterations, torch::Device device) {
    model->to(device); model->train();
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(3e-4));
    double seconds = 0.0, loss_value = 0.0;
    for (int64_t i = 0; i < warmup + iterations; ++i) {
        if (device.is_cuda()) torch::cuda::synchronize();
        const auto start = std::chrono::steady_clock::now();
        optimizer.zero_grad();
        auto logits = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, cfg.vocab_size}), target.reshape({-1}));
        loss.backward(); optimizer.step();
        if (device.is_cuda()) torch::cuda::synchronize();
        const auto end = std::chrono::steady_clock::now();
        loss_value = loss.template item<double>();
        if (i >= warmup) seconds += std::chrono::duration<double>(end - start).count();
    }
    const auto all_params = parameter_count(*model);
    const auto expert_params = parameter_count(*model, true);
    const auto active_params = label == "NEXUS hybrid"
        ? all_params - expert_params + expert_params * cfg.top_k / cfg.experts
        : all_params;
    const double tokens = static_cast<double>(input.size(0) * input.size(1) * iterations);
    std::cout << label << '\n'
              << "  stored_parameters=" << all_params << '\n'
              << "  estimated_active_parameters_per_token=" << active_params << '\n'
              << "  average_train_step_ms=" << (seconds * 1000.0 / iterations) << '\n'
              << "  training_tokens_per_second=" << (tokens / seconds) << '\n'
              << "  final_random_batch_loss=" << loss_value << '\n';
}
}
int main(int argc, char** argv) {
    try {
        const int64_t sequence = argc > 1 ? std::stoll(argv[1]) : 128;
        const int64_t iterations = argc > 2 ? std::stoll(argv[2]) : 10;
        if (sequence < 2 || iterations < 1) throw std::invalid_argument("sequence must be >=2 and iterations >=1");
        nexus::ModelConfig cfg; cfg.context = sequence;
        auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        torch::manual_seed(2026);
        auto input = torch::randint(0, cfg.vocab_size, {2, sequence}, torch::TensorOptions().dtype(torch::kInt64)).to(device);
        auto target = torch::randint(0, cfg.vocab_size, {2, sequence}, torch::TensorOptions().dtype(torch::kInt64)).to(device);
        nexus::NexusModel hybrid(cfg);
        // Width 9 SwiGLU is a rough active-parameter match for two active experts,
        // the SSM path and attention on every other NEXUS block. Stored capacity differs.
        nexus::TransformerBaseline baseline(cfg, 9);
        std::cout << "device=" << (device.is_cuda() ? "CUDA" : "CPU")
                  << " batch=2 sequence=" << sequence << " warmup=2 iterations=" << iterations << '\n';
        run_benchmark("NEXUS hybrid", hybrid, cfg, input, target, 2, iterations, device);
        run_benchmark("Dense Transformer reference (SwiGLU x9)", baseline, cfg, input, target, 2, iterations, device);
        std::cout << "Note: one random-batch microbenchmark is not a quality benchmark or a universal speed claim.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n'; return 1;
    }
}
