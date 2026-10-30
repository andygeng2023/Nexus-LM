#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace nexus {
class ByteTokenizer {
public:
 static constexpr int64_t vocab_size=256;
 std::vector<int64_t> encode(const std::string& text) const;
 std::string decode(const std::vector<int64_t>& ids) const;
};
}
