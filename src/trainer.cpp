#include "nexus/trainer.h"
#include "nexus/tokenizer.h"
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
namespace nexus {
Trainer::Trainer(NexusModel m,TrainConfig c,torch::Device d):model_(std::move(m)),config_(c),device_(d){model_->to(device_);}
void Trainer::train_text_file(const std::filesystem::path& path,const std::filesystem::path& checkpoint){
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open corpus: "+path.string());
 std::string text((std::istreambuf_iterator<char>(f)),{});ByteTokenizer tok;auto ids=tok.encode(text);
 if(config_.steps<1||config_.batch_size<1||config_.sequence_length<1||ids.size()<=static_cast<size_t>(config_.sequence_length+1))throw std::runtime_error("invalid training settings or corpus too short");
 auto data=torch::tensor(ids,torch::kInt64).to(device_);torch::manual_seed(config_.seed);
 torch::optim::AdamW opt(model_->parameters(),torch::optim::AdamWOptions(config_.learning_rate));std::mt19937_64 rng(config_.seed);
 std::uniform_int_distribution<int64_t> dist(0,static_cast<int64_t>(ids.size())-config_.sequence_length-2);model_->train();
 for(int64_t step=1;step<=config_.steps;++step){std::vector<torch::Tensor>x,y;
 for(int64_t b=0;b<config_.batch_size;++b){auto s=dist(rng);x.push_back(data.slice(0,s,s+config_.sequence_length));y.push_back(data.slice(0,s+1,s+config_.sequence_length+1));}
 opt.zero_grad();auto logits=model_->forward(torch::stack(x));auto loss=torch::nn::functional::cross_entropy(logits.reshape({-1,model_->config().vocab_size}),torch::stack(y).reshape({-1}));
 loss.backward();torch::nn::utils::clip_grad_norm_(model_->parameters(),1.0);opt.step();
 if(step==1||(config_.log_every>0&&step%config_.log_every==0))std::cout<<"step "<<step<<"/"<<config_.steps<<" loss="<<loss.item<double>()<<'\n';
 if(config_.save_every>0&&step%config_.save_every==0)torch::save(model_,checkpoint);}
 torch::save(model_,checkpoint);std::cout<<"checkpoint saved: "<<checkpoint<<'\n';
}
}
