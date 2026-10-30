#include "nexus/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace nexus {
namespace {
struct FeedForwardImpl:torch::nn::Module {
 explicit FeedForwardImpl(int64_t w){up=register_module("up",torch::nn::Linear(w,w*4));gate=register_module("gate",torch::nn::Linear(w,w*4));down=register_module("down",torch::nn::Linear(w*4,w));}
 torch::Tensor forward(torch::Tensor x){return down(torch::silu(gate(x))*up(x));}
 torch::nn::Linear up{nullptr},gate{nullptr},down{nullptr};
};
}
HybridBlockImpl::HybridBlockImpl(const ModelConfig& c,int64_t i):cfg_(c),use_attention_(i%2==1){
 if(c.width%c.heads) throw std::invalid_argument("width must be divisible by heads");
 norm_state_=register_module("norm_state",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.width})));
 state_conv_=register_module("state_conv",torch::nn::Conv1d(torch::nn::Conv1dOptions(c.width,c.width,5).padding(4).groups(c.width)));
 state_gate_=register_module("state_gate",torch::nn::Linear(c.width,c.width));
 if(use_attention_){norm_attention_=register_module("norm_attention",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.width})));qkv_=register_module("qkv",torch::nn::Linear(c.width,c.width*3));attention_out_=register_module("attention_out",torch::nn::Linear(c.width,c.width));}
 norm_moe_=register_module("norm_moe",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.width})));
 router_=register_module("router",torch::nn::Linear(c.width,c.experts)); experts_=register_module("experts",torch::nn::ModuleList());
 for(int64_t e=0;e<c.experts;++e) experts_->push_back(std::make_shared<FeedForwardImpl>(c.width));
}
torch::Tensor HybridBlockImpl::forward(torch::Tensor x){
 auto s=norm_state_(x); auto conv=state_conv_(s.transpose(1,2)); conv=conv.narrow(2,0,s.size(1)).transpose(1,2);
 x=x+torch::tanh(state_gate_(s))*torch::silu(conv);
 if(use_attention_){auto a=norm_attention_(x);auto b=a.size(0),t=a.size(1),d=cfg_.width/cfg_.heads;
 auto qkv=qkv_(a).view({b,t,3,cfg_.heads,d}).permute({2,0,3,1,4});auto q=qkv[0],k=qkv[1],v=qkv[2];
 auto scores=torch::matmul(q,k.transpose(-2,-1))/std::sqrt(static_cast<double>(d));
 auto mask=torch::ones({t,t},scores.options()).triu(1).to(torch::kBool);scores=scores.masked_fill(mask,-1.0e9);
 auto weights=torch::softmax(scores,-1);if(cfg_.dropout>0&&is_training())weights=torch::dropout(weights,cfg_.dropout,true);
 auto out=torch::matmul(weights,v).transpose(1,2).contiguous().view({b,t,cfg_.width});x=x+attention_out_(out);}
 auto n=norm_moe_(x);auto flat=n.reshape({-1,cfg_.width});auto top=torch::topk(router_(flat),std::min(cfg_.top_k,cfg_.experts),-1);
 auto ids=std::get<1>(top);auto weights=torch::softmax(std::get<0>(top),-1);auto output=torch::zeros_like(flat);
 for(int64_t e=0;e<cfg_.experts;++e){auto loc=torch::nonzero(ids==e);if(!loc.numel())continue;auto ti=loc.select(1,0),si=loc.select(1,1);
 auto inp=flat.index_select(0,ti);auto expert=std::dynamic_pointer_cast<FeedForwardImpl>(experts_[e]);
 auto w=weights.index({ti,si}).unsqueeze(1);output.index_add_(0,ti,expert->forward(inp)*w);}
 return x+output.view_as(x);
}
NexusModelImpl::NexusModelImpl(const ModelConfig& c):cfg_(c){
 if(c.layers<1||c.context<2||c.experts<1||c.top_k<1||c.top_k>c.experts)throw std::invalid_argument("invalid model configuration");
 embedding_=register_module("embedding",torch::nn::Embedding(c.vocab_size,c.width));blocks_=register_module("blocks",torch::nn::ModuleList());
 for(int64_t i=0;i<c.layers;++i)blocks_->push_back(HybridBlock(c,i));
 final_norm_=register_module("final_norm",torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.width})));
 lm_head_=register_module("lm_head",torch::nn::Linear(c.width,c.vocab_size));lm_head_->weight=embedding_->weight;
}
torch::Tensor NexusModelImpl::forward(torch::Tensor tokens){
 if(tokens.dim()!=2)throw std::invalid_argument("tokens must have shape [batch, sequence]");
 if(tokens.size(1)>cfg_.context)throw std::invalid_argument("sequence exceeds configured context");
 auto x=embedding_(tokens);for(auto& m:*blocks_)x=m->as<HybridBlockImpl>()->forward(x);return lm_head_(final_norm_(x));
}
torch::Tensor NexusModelImpl::generate(torch::Tensor prompt,int64_t count,double temp,int64_t k){
 if(temp<=0)throw std::invalid_argument("temperature must be positive");eval();torch::NoGradGuard guard;auto out=prompt;
 for(int64_t i=0;i<count;++i){auto in=out.size(1)>cfg_.context?out.slice(1,out.size(1)-cfg_.context):out;
 auto logits=forward(in).select(1,in.size(1)-1)/temp;if(k>0&&k<logits.size(-1)){auto v=std::get<0>(torch::topk(logits,k,-1));logits=logits.masked_fill(logits<v.select(-1,k-1).unsqueeze(-1),-1e9);}
 out=torch::cat({out,torch::multinomial(torch::softmax(logits,-1),1)},1);}return out;
}
}
