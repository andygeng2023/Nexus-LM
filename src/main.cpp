#include "nexus/model.h"
#include "nexus/tokenizer.h"
#include "nexus/trainer.h"
#include <torch/torch.h>
#include <iostream>
int main(int argc,char** argv){try{
 if(argc<2){std::cout<<"NEXUS-LM 0.1.0\n  nexus train <text-file> <checkpoint> [steps]\n  nexus generate <checkpoint> <prompt> [new-tokens]\n";return 0;}
 nexus::ModelConfig mc;auto device=torch::cuda::is_available()?torch::Device(torch::kCUDA):torch::Device(torch::kCPU);
 std::cout<<"device: "<<(device.is_cuda()?"CUDA":"CPU")<<'\n';std::string cmd=argv[1];
 if(cmd=="train"){if(argc<4)return 2;nexus::TrainConfig tc;if(argc>4)tc.steps=std::stoll(argv[4]);nexus::NexusModel model(mc);nexus::Trainer(model,tc,device).train_text_file(argv[2],argv[3]);return 0;}
 if(cmd=="generate"){if(argc<4)return 2;nexus::NexusModel model(mc);torch::load(model,argv[2]);model->to(device);nexus::ByteTokenizer tok;auto ids=tok.encode(argv[3]);
 if(ids.empty())throw std::runtime_error("prompt must not be empty");auto input=torch::tensor(ids,torch::kInt64).unsqueeze(0).to(device);auto n=argc>4?std::stoll(argv[4]):80;
 auto result=model->generate(input,n).to(torch::kCPU).squeeze(0).contiguous();std::vector<int64_t> out(result.data_ptr<int64_t>(),result.data_ptr<int64_t>()+result.numel());std::cout<<tok.decode(out)<<'\n';return 0;}
 std::cerr<<"unknown command\n";return 2;
 }catch(const std::exception& e){std::cerr<<"NEXUS error: "<<e.what()<<'\n';return 1;}}
