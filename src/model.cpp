#include "nexus/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nexus {
namespace {
struct FeedForwardImpl : torch::nn::Module {
    explicit FeedForwardImpl(int64_t width) {
        up = register_module("up", torch::nn::Linear(width, width * 4));
        gate = register_module("gate", torch::nn::Linear(width, width * 4));
        down = register_module("down", torch::nn::Linear(width * 4, width));
    }
    torch::Tensor forward(const torch::Tensor& x) { return down(torch::silu(gate(x)) * up(x)); }
    torch::nn::Linear up{nullptr}, gate{nullptr}, down{nullptr};
};
}

SelectiveStateSpaceImpl::SelectiveStateSpaceImpl(int64_t width) : width_(width) {
    if (width < 1) throw std::invalid_argument("state-space width must be positive");
    parameters_ = register_module("parameters", torch::nn::Linear(width, width * 3));
    output_ = register_module("output", torch::nn::Linear(width, width));
    auto positions = torch::arange(1, width + 1, torch::TensorOptions().dtype(torch::kFloat32));
    A_log_ = register_parameter("A_log", torch::log(positions));
    D_ = register_parameter("D", torch::ones({width}, torch::TensorOptions().dtype(torch::kFloat32)));
}
torch::Tensor SelectiveStateSpaceImpl::forward(const torch::Tensor& x) {
    if (x.dim() != 3 || x.size(2) != width_) throw std::invalid_argument("SSM input must have shape [batch, sequence, width]");
    const auto batch = x.size(0);
    const auto steps = x.size(1);
    auto projected = parameters_(x);
    auto parts = projected.chunk(3, -1);
    auto delta = torch::softplus(parts[0]) + 1.0e-4;
    auto input_gate = torch::tanh(parts[1]);
    auto output_gate = torch::tanh(parts[2]);
    auto decay_rate = torch::exp(A_log_).view({1, width_});
    auto state = torch::zeros({batch, width_}, x.options());
    std::vector<torch::Tensor> outputs;
    outputs.reserve(static_cast<std::size_t>(steps));
    for (int64_t t = 0; t < steps; ++t) {
        auto xt = x.select(1, t);
        auto dt = delta.select(1, t);
        auto decay = torch::exp(-dt * decay_rate);
        auto b = input_gate.select(1, t);
        auto c = output_gate.select(1, t);
        state = decay * state + (1.0 - decay) * b * xt;
        outputs.push_back(state * c + D_ * xt);
    }
    if (outputs.empty()) return x;
    return output_(torch::stack(outputs, 1));
}

HybridBlockImpl::HybridBlockImpl(const ModelConfig& cfg, int64_t layer_index)
    : cfg_(cfg), use_attention_(layer_index % 2 == 1) {
    if (cfg.width % cfg.heads != 0) throw std::invalid_argument("model width must be divisible by attention heads");
    norm_ssm_ = register_module("norm_ssm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.width})));
    ssm_ = register_module("selective_ssm", SelectiveStateSpace(cfg.width));
    if (use_attention_) {
        norm_attention_ = register_module("norm_attention", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.width})));
        qkv_ = register_module("qkv", torch::nn::Linear(cfg.width, cfg.width * 3));
        attention_out_ = register_module("attention_out", torch::nn::Linear(cfg.width, cfg.width));
    }
    norm_moe_ = register_module("norm_moe", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.width})));
    router_ = register_module("router", torch::nn::Linear(cfg.width, cfg.experts));
    experts_ = register_module("experts", torch::nn::ModuleList());
    for (int64_t e = 0; e < cfg.experts; ++e) experts_->push_back(std::make_shared<FeedForwardImpl>(cfg.width));
}
torch::Tensor HybridBlockImpl::forward(torch::Tensor x) {
    x = x + ssm_->forward(norm_ssm_(x));
    if (use_attention_) {
        auto a = norm_attention_(x);
        const auto batch = a.size(0), time = a.size(1), head_dim = cfg_.width / cfg_.heads;
        auto qkv = qkv_(a).view({batch, time, 3, cfg_.heads, head_dim}).permute({2, 0, 3, 1, 4});
        auto q = qkv[0], k = qkv[1], v = qkv[2];
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
        auto causal = torch::ones({time, time}, scores.options()).triu(1).to(torch::kBool);
        scores = scores.masked_fill(causal, -1.0e9);
        auto weights = torch::softmax(scores, -1);
        if (cfg_.dropout > 0.0 && is_training()) weights = torch::dropout(weights, cfg_.dropout, true);
        auto attended = torch::matmul(weights, v).transpose(1, 2).contiguous().view({batch, time, cfg_.width});
        x = x + attention_out_(attended);
    }

    auto normalized = norm_moe_(x);
    auto flat = normalized.reshape({-1, cfg_.width});
    const auto k = std::min(cfg_.top_k, cfg_.experts);
    auto routed = torch::topk(router_(flat), k, -1);
    auto route_weights = torch::softmax(std::get<0>(routed), -1);
    auto route_ids = std::get<1>(routed);
    auto mixed = torch::zeros_like(flat);
    for (int64_t expert_id = 0; expert_id < cfg_.experts; ++expert_id) {
        auto locations = torch::nonzero(route_ids == expert_id);
        if (locations.numel() == 0) continue;
        auto token_indices = locations.select(1, 0);
        auto choice_indices = locations.select(1, 1);
        auto expert_input = flat.index_select(0, token_indices);
        auto expert = std::dynamic_pointer_cast<FeedForwardImpl>(experts_[expert_id]);
        auto weights = route_weights.index({token_indices, choice_indices}).unsqueeze(1);
        mixed.index_add_(0, token_indices, expert->forward(expert_input) * weights);
    }
    return x + mixed.view_as(x);
}

NexusModelImpl::NexusModelImpl(const ModelConfig& cfg) : cfg_(cfg) {
    if (cfg.vocab_size < 256 || cfg.width < 1 || cfg.layers < 1 || cfg.heads < 1 ||
        cfg.context < 2 || cfg.experts < 1 || cfg.top_k < 1 || cfg.top_k > cfg.experts || cfg.width % cfg.heads != 0)
        throw std::invalid_argument("invalid NEXUS model configuration");
    embedding_ = register_module("embedding", torch::nn::Embedding(cfg.vocab_size, cfg.width));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.layers; ++i) blocks_->push_back(HybridBlock(cfg, i));
    final_norm_ = register_module("final_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.width})));
    lm_head_ = register_module("lm_head", torch::nn::Linear(cfg.width, cfg.vocab_size));
    lm_head_->weight = embedding_->weight;
}
torch::Tensor NexusModelImpl::forward(torch::Tensor tokens) {
    if (tokens.dim() != 2) throw std::invalid_argument("tokens must have shape [batch, sequence]");
    if (tokens.size(1) < 1 || tokens.size(1) > cfg_.context) throw std::invalid_argument("sequence length outside configured context");
    auto x = embedding_(tokens);
    for (auto& block : *blocks_) x = block->as<HybridBlockImpl>()->forward(x);
    return lm_head_(final_norm_(x));
}
torch::Tensor NexusModelImpl::generate(torch::Tensor prompt, int64_t new_tokens, double temperature, int64_t top_k) {
    if (prompt.dim() != 2 || prompt.size(1) == 0) throw std::invalid_argument("prompt must have shape [batch, non-empty sequence]");
    if (temperature <= 0.0 || new_tokens < 0) throw std::invalid_argument("temperature must be positive and token count non-negative");
    eval(); torch::NoGradGuard guard; auto output = prompt;
    for (int64_t i = 0; i < new_tokens; ++i) {
        auto input = output.size(1) > cfg_.context ? output.slice(1, output.size(1) - cfg_.context) : output;
        auto logits = forward(input).select(1, input.size(1) - 1) / temperature;
        if (top_k > 0 && top_k < logits.size(-1)) {
            auto threshold = std::get<0>(torch::topk(logits, top_k, -1)).select(-1, top_k - 1).unsqueeze(-1);
            logits = logits.masked_fill(logits < threshold, -1.0e9);
        }
        output = torch::cat({output, torch::multinomial(torch::softmax(logits, -1), 1)}, 1);
    }
    return output;
}

TransformerBlockImpl::TransformerBlockImpl(int64_t width, int64_t heads, int64_t hidden, double dropout)
    : width_(width), heads_(heads), dropout_(dropout) {
    if (width % heads != 0) throw std::invalid_argument("Transformer width must be divisible by heads");
    norm_attention_ = register_module("norm_attention", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})));
    norm_ffn_ = register_module("norm_ffn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({width})));
    qkv_ = register_module("qkv", torch::nn::Linear(width, width * 3));
    attention_out_ = register_module("attention_out", torch::nn::Linear(width, width));
    up_ = register_module("up", torch::nn::Linear(width, hidden));
    gate_ = register_module("gate", torch::nn::Linear(width, hidden));
    down_ = register_module("down", torch::nn::Linear(hidden, width));
}
torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
    auto a = norm_attention_(x);
    const auto batch = a.size(0), time = a.size(1), head_dim = width_ / heads_;
    auto qkv = qkv_(a).view({batch, time, 3, heads_, head_dim}).permute({2, 0, 3, 1, 4});
    auto q = qkv[0], k = qkv[1], v = qkv[2];
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    auto causal = torch::ones({time, time}, scores.options()).triu(1).to(torch::kBool);
    scores = scores.masked_fill(causal, -1.0e9);
    auto weights = torch::softmax(scores, -1);
    if (dropout_ > 0.0 && is_training()) weights = torch::dropout(weights, dropout_, true);
    auto attended = torch::matmul(weights, v).transpose(1, 2).contiguous().view({batch, time, width_});
    x = x + attention_out_(attended);
    auto f = norm_ffn_(x);
    return x + down_(torch::silu(gate_(f)) * up_(f));
}
TransformerBaselineImpl::TransformerBaselineImpl(const ModelConfig& cfg, int64_t ffn_multiplier) : cfg_(cfg) {
    if (ffn_multiplier < 1 || cfg.width % cfg.heads != 0) throw std::invalid_argument("invalid Transformer baseline configuration");
    embedding_ = register_module("embedding", torch::nn::Embedding(cfg.vocab_size, cfg.width));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.layers; ++i)
        blocks_->push_back(TransformerBlock(cfg.width, cfg.heads, cfg.width * ffn_multiplier, cfg.dropout));
    final_norm_ = register_module("final_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.width})));
    lm_head_ = register_module("lm_head", torch::nn::Linear(cfg.width, cfg.vocab_size));
    lm_head_->weight = embedding_->weight;
}
torch::Tensor TransformerBaselineImpl::forward(torch::Tensor tokens) {
    if (tokens.dim() != 2 || tokens.size(1) < 1 || tokens.size(1) > cfg_.context)
        throw std::invalid_argument("Transformer input must be [batch, sequence] within context");
    auto x = embedding_(tokens);
    for (auto& block : *blocks_) x = block->as<TransformerBlockImpl>()->forward(x);
    return lm_head_(final_norm_(x));
}
}
