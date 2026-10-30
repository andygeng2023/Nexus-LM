#pragma once
#include <torch/torch.h>
namespace nexus {
struct ModelConfig {
 int64_t vocab_size=256,width=128,layers=4,heads=4,context=256,experts=4,top_k=2;
 double dropout=0.0;
};
struct HybridBlockImpl:torch::nn::Module {
 explicit HybridBlockImpl(const ModelConfig&,int64_t);
 torch::Tensor forward(torch::Tensor);
private:
 ModelConfig cfg_; bool use_attention_;
 torch::nn::LayerNorm norm_state_{nullptr},norm_attention_{nullptr},norm_moe_{nullptr};
 torch::nn::Conv1d state_conv_{nullptr};
 torch::nn::Linear state_gate_{nullptr},qkv_{nullptr},attention_out_{nullptr},router_{nullptr};
 torch::nn::ModuleList experts_{nullptr};
};
TORCH_MODULE(HybridBlock);
struct NexusModelImpl:torch::nn::Module {
 explicit NexusModelImpl(const ModelConfig&);
 torch::Tensor forward(torch::Tensor);
 torch::Tensor generate(torch::Tensor,int64_t,double temperature=0.8,int64_t top_k=40);
 const ModelConfig& config() const{return cfg_;}
private:
 ModelConfig cfg_; torch::nn::Embedding embedding_{nullptr}; torch::nn::ModuleList blocks_{nullptr};
 torch::nn::LayerNorm final_norm_{nullptr}; torch::nn::Linear lm_head_{nullptr};
};
TORCH_MODULE(NexusModel);
}
