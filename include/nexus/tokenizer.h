#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
class ByteTokenizer {
public:
    static constexpr int64_t vocab_size = 256;
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;
};

// Reversible byte-level BPE. It never splits UTF-8 bytes and always decodes
// back to the original byte sequence. The initial 256 entries are raw bytes.
class BpeTokenizer {
public:
    BpeTokenizer();
    void train(const std::string& corpus, int64_t target_vocab_size);
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);
    int64_t vocab_size() const noexcept;
    std::size_t merge_count() const noexcept;
private:
    struct Merge { int64_t left; int64_t right; };
    std::vector<std::string> tokens_;
    std::vector<Merge> merges_;
};
}
