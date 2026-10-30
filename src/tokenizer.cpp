#include "nexus/tokenizer.h"
#include <stdexcept>
namespace nexus {
std::vector<int64_t> ByteTokenizer::encode(const std::string& text) const {
 std::vector<int64_t> ids; ids.reserve(text.size());
 for(unsigned char c:text) ids.push_back(static_cast<int64_t>(c));
 return ids;
}
std::string ByteTokenizer::decode(const std::vector<int64_t>& ids) const {
 std::string text; text.reserve(ids.size());
 for(auto id:ids){ if(id<0||id>=vocab_size) throw std::out_of_range("token id outside byte vocabulary"); text.push_back(static_cast<char>(static_cast<unsigned char>(id))); }
 return text;
}
}
