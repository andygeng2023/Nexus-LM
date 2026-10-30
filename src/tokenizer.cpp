#include "nexus/tokenizer.h"
#include <fstream>
#include <map>
#include <stdexcept>
#include <iterator>

namespace nexus {
std::vector<int64_t> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids; ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back(static_cast<int64_t>(c));
    return ids;
}
std::string ByteTokenizer::decode(const std::vector<int64_t>& ids) const {
    std::string text; text.reserve(ids.size());
    for (const auto id : ids) {
        if (id < 0 || id >= vocab_size) throw std::out_of_range("byte token id out of range");
        text.push_back(static_cast<char>(static_cast<unsigned char>(id)));
    }
    return text;
}

BpeTokenizer::BpeTokenizer() {
    tokens_.reserve(256);
    for (int i = 0; i < 256; ++i) tokens_.emplace_back(1, static_cast<char>(i));
}
void BpeTokenizer::train(const std::string& corpus, int64_t target_vocab_size) {
    if (target_vocab_size < 256) throw std::invalid_argument("BPE vocabulary must be at least 256");
    tokens_.clear(); merges_.clear();
    for (int i = 0; i < 256; ++i) tokens_.emplace_back(1, static_cast<char>(i));
    std::vector<int64_t> sequence; sequence.reserve(corpus.size());
    for (unsigned char c : corpus) sequence.push_back(static_cast<int64_t>(c));

    while (static_cast<int64_t>(tokens_.size()) < target_vocab_size && sequence.size() > 1) {
        std::map<std::pair<int64_t, int64_t>, std::size_t> frequencies;
        for (std::size_t i = 0; i + 1 < sequence.size(); ++i)
            ++frequencies[{sequence[i], sequence[i + 1]}];
        if (frequencies.empty()) break;
        auto best = frequencies.begin();
        for (auto it = frequencies.begin(); it != frequencies.end(); ++it)
            if (it->second > best->second) best = it;
        if (best->second < 2) break;

        const auto left = best->first.first;
        const auto right = best->first.second;
        const int64_t new_id = static_cast<int64_t>(tokens_.size());
        tokens_.push_back(tokens_.at(static_cast<std::size_t>(left)) + tokens_.at(static_cast<std::size_t>(right)));
        merges_.push_back({left, right});

        std::vector<int64_t> merged; merged.reserve(sequence.size());
        for (std::size_t i = 0; i < sequence.size();) {
            if (i + 1 < sequence.size() && sequence[i] == left && sequence[i + 1] == right) {
                merged.push_back(new_id); i += 2;
            } else { merged.push_back(sequence[i]); ++i; }
        }
        sequence.swap(merged);
    }
}
std::vector<int64_t> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> ids; ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back(static_cast<int64_t>(c));
    for (std::size_t rank = 0; rank < merges_.size(); ++rank) {
        const auto left = merges_[rank].left, right = merges_[rank].right;
        const auto new_id = static_cast<int64_t>(256 + rank);
        std::vector<int64_t> merged; merged.reserve(ids.size());
        for (std::size_t i = 0; i < ids.size();) {
            if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
                merged.push_back(new_id); i += 2;
            } else { merged.push_back(ids[i]); ++i; }
        }
        ids.swap(merged);
    }
    return ids;
}
std::string BpeTokenizer::decode(const std::vector<int64_t>& ids) const {
    std::string text;
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size())
            throw std::out_of_range("BPE token id out of range");
        text += tokens_[static_cast<std::size_t>(id)];
    }
    return text;
}
void BpeTokenizer::save(const std::filesystem::path& path) const {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write tokenizer: " + path.string());
    out << "NEXUS_BPE_V1\n" << tokens_.size() << '\n';
    for (const auto& merge : merges_) out << merge.left << ' ' << merge.right << '\n';
    if (!out) throw std::runtime_error("failed while writing tokenizer: " + path.string());
}
void BpeTokenizer::load(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read tokenizer: " + path.string());
    std::string magic; std::getline(in, magic);
    if (magic != "NEXUS_BPE_V1") throw std::runtime_error("unsupported tokenizer file format");
    std::size_t count = 0;
    if (!(in >> count) || count < 256) throw std::runtime_error("invalid tokenizer vocabulary size");
    std::vector<std::string> new_tokens;
    std::vector<Merge> new_merges;
    new_tokens.reserve(count); new_merges.reserve(count - 256);
    for (int i = 0; i < 256; ++i) new_tokens.emplace_back(1, static_cast<char>(i));
    for (std::size_t id = 256; id < count; ++id) {
        int64_t left = -1, right = -1;
        if (!(in >> left >> right) || left < 0 || right < 0 ||
            static_cast<std::size_t>(left) >= new_tokens.size() ||
            static_cast<std::size_t>(right) >= new_tokens.size())
            throw std::runtime_error("invalid BPE merge table");
        new_tokens.push_back(new_tokens[static_cast<std::size_t>(left)] + new_tokens[static_cast<std::size_t>(right)]);
        new_merges.push_back({left, right});
    }
    tokens_.swap(new_tokens); merges_.swap(new_merges);
}
int64_t BpeTokenizer::vocab_size() const noexcept { return static_cast<int64_t>(tokens_.size()); }
std::size_t BpeTokenizer::merge_count() const noexcept { return merges_.size(); }
}
