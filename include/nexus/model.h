#pragma once
#include <torch/torch.h>

namespace nexus {
struct ModelConfig {
    int64_t vocab_size = 512;
    int64_t width = 128;
    int64_t layers = 4;
    int64_t heads = 4;
    int64_t context = 256;
    int64_t experts = 4;
    int64_t top_k = 2;
    double dropout = 0.0;
};

// Input-selective diagonal state-space layer. Each token predicts its own
// discretisation step and B/C vectors; state is recurrent and strictly causal.
struct SelectiveStateSpaceImpl : torch::nn::Module {
    explicit SelectiveStateSpaceImpl(int64_t width);
    torch::Tensor forward(const torch::Tensor& x);
private:
    int64_t width_;
    torch::nn::Linear parameters_{nullptr}, output_{nullptr};
    torch::Tensor A_log_, D_;
};
TORCH_MODULE(SelectiveStateSpace);

struct HybridBlockImpl : torch::nn::Module {
    explicit HybridBlockImpl(const ModelConfig&, int64_t layer_index);
    torch::Tensor forward(torch::Tensor x);
private:
    ModelConfig cfg_;
    bool use_attention_;
    torch::nn::LayerNorm norm_ssm_{nullptr}, norm_attention_{nullptr}, norm_moe_{nullptr};
    SelectiveStateSpace ssm_{nullptr};
    torch::nn::Linear qkv_{nullptr}, attention_out_{nullptr}, router_{nullptr};
    torch::nn::ModuleList experts_{nullptr};
};
TORCH_MODULE(HybridBlock);

struct NexusModelImpl : torch::nn::Module {
    explicit NexusModelImpl(const ModelConfig&);
    torch::Tensor forward(torch::Tensor tokens);
    torch::Tensor generate(torch::Tensor prompt, int64_t new_tokens,
                           double temperature = 0.8, int64_t top_k = 40);
    const ModelConfig& config() const noexcept { return cfg_; }
private:
    ModelConfig cfg_;
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
    torch::nn::Linear lm_head_{nullptr};
};
TORCH_MODULE(NexusModel);

// Dense causal Transformer reference with a widened SwiGLU MLP. The default
// multiplier approximates NEXUS's active (not stored) per-token parameter budget.
struct TransformerBlockImpl : torch::nn::Module {
    TransformerBlockImpl(int64_t width, int64_t heads, int64_t hidden, double dropout);
    torch::Tensor forward(torch::Tensor x);
private:
    int64_t width_, heads_;
    double dropout_;
    torch::nn::LayerNorm norm_attention_{nullptr}, norm_ffn_{nullptr};
    torch::nn::Linear qkv_{nullptr}, attention_out_{nullptr}, up_{nullptr}, gate_{nullptr}, down_{nullptr};
};
TORCH_MODULE(TransformerBlock);
struct TransformerBaselineImpl : torch::nn::Module {
    explicit TransformerBaselineImpl(const ModelConfig&, int64_t ffn_multiplier = 9);
    torch::Tensor forward(torch::Tensor tokens);
private:
    ModelConfig cfg_;
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
    torch::nn::Linear lm_head_{nullptr};
};
TORCH_MODULE(TransformerBaseline);
}
