#include "nexus/tokenizer.h"
#include "nexus/model.h"
#include <cassert>
#include <iostream>
int main(){nexus::ByteTokenizer t;std::string s="NEXUS LM: local model";assert(t.decode(t.encode(s))==s);
 nexus::ModelConfig c;c.width=32;c.layers=2;c.heads=4;c.context=16;c.experts=2;c.top_k=1;nexus::NexusModel m(c);
 auto x=torch::randint(0,c.vocab_size,{2,12},torch::TensorOptions().dtype(torch::kInt64));auto y=m->forward(x);assert(y.sizes()==torch::IntArrayRef({2,12,c.vocab_size}));
 auto loss=torch::nn::functional::cross_entropy(y.reshape({-1,c.vocab_size}),x.reshape({-1}));loss.backward();assert(loss.item<double>()>0);std::cout<<"NEXUS core tests passed\n";}
