#include "nexus/tokenizer.h"
#include <cstdint>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <unordered_map>

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
    if (corpus.empty()) return;

    const std::size_t n = corpus.size();
    std::vector<int64_t> values(n), previous(n), next(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<unsigned char>(corpus[i]);
        previous[i] = i == 0 ? -1 : static_cast<int64_t>(i - 1);
        next[i] = i + 1 == n ? -1 : static_cast<int64_t>(i + 1);
    }
    using PairKey = uint64_t;
    auto pair_key = [](int64_t left, int64_t right) -> PairKey {
        return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
    };
    struct PairState { int64_t count = 0; uint64_t version = 0; };
    struct HeapItem { int64_t count; PairKey key; uint64_t version; };
    struct LowerPriority {
        bool operator()(const HeapItem& a, const HeapItem& b) const {
            if (a.count != b.count) return a.count < b.count;
            return a.key > b.key;
        }
    };
    std::unordered_map<PairKey, PairState> frequencies;
    frequencies.reserve(n / 2 + 1);
    std::priority_queue<HeapItem, std::vector<HeapItem>, LowerPriority> queue;
    auto adjust = [&](PairKey key, int64_t delta) {
        auto& state = frequencies[key];
        state.count += delta;
        ++state.version;
        if (state.count > 0) queue.push({state.count, key, state.version});
    };
    for (std::size_t i = 0; i + 1 < n; ++i) adjust(pair_key(values[i], values[i + 1]), 1);

    while (static_cast<int64_t>(tokens_.size()) < target_vocab_size && !queue.empty()) {
        const auto top = queue.top(); queue.pop();
        const auto found = frequencies.find(top.key);
        if (found == frequencies.end() || found->second.version != top.version || found->second.count != top.count)
            continue;
        if (top.count < 2) break;
        const auto left = static_cast<int64_t>(top.key >> 32);
        const auto right = static_cast<int64_t>(top.key & 0xffffffffULL);
        const int64_t new_id = static_cast<int64_t>(tokens_.size());
        tokens_.push_back(tokens_.at(static_cast<std::size_t>(left)) + tokens_.at(static_cast<std::size_t>(right)));
        merges_.push_back({left, right});

        int64_t current = 0;
        while (current >= 0) {
            const auto rhs = next[static_cast<std::size_t>(current)];
            if (rhs >= 0 && values[static_cast<std::size_t>(current)] == left &&
                values[static_cast<std::size_t>(rhs)] == right) {
                const auto lhs = previous[static_cast<std::size_t>(current)];
                const auto after = next[static_cast<std::size_t>(rhs)];
                if (lhs >= 0) {
                    adjust(pair_key(values[static_cast<std::size_t>(lhs)], left), -1);
                    adjust(pair_key(values[static_cast<std::size_t>(lhs)], new_id), 1);
                }
                adjust(pair_key(left, right), -1);
                if (after >= 0) {
                    adjust(pair_key(right, values[static_cast<std::size_t>(after)]), -1);
                    adjust(pair_key(new_id, values[static_cast<std::size_t>(after)]), 1);
                    previous[static_cast<std::size_t>(after)] = current;
                }
                values[static_cast<std::size_t>(current)] = new_id;
                next[static_cast<std::size_t>(current)] = after;
                previous[static_cast<std::size_t>(rhs)] = -1;
                next[static_cast<std::size_t>(rhs)] = -1;
                current = after;
            } else {
                current = rhs;
            }
        }
    }
}
std::vector<int64_t> BpeTokenizer::encode(const std::string& text) const {
    const std::size_t n = text.size();
    if (n == 0) return {};
    using PairKey = uint64_t;
    auto pair_key = [](int64_t left, int64_t right) -> PairKey {
        return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
    };
    std::unordered_map<PairKey, int64_t> merge_rank;
    merge_rank.reserve(merges_.size() * 2 + 1);
    for (std::size_t i = 0; i < merges_.size(); ++i)
        merge_rank.emplace(pair_key(merges_[i].left, merges_[i].right), static_cast<int64_t>(i));

    std::vector<int64_t> values(n), previous(n), next(n);
    std::vector<uint64_t> versions(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<unsigned char>(text[i]);
        previous[i] = i == 0 ? -1 : static_cast<int64_t>(i - 1);
        next[i] = i + 1 == n ? -1 : static_cast<int64_t>(i + 1);
    }
    struct Candidate { int64_t rank; int64_t position; uint64_t version; };
    struct LaterCandidate {
        bool operator()(const Candidate& a, const Candidate& b) const {
            if (a.rank != b.rank) return a.rank > b.rank;
            return a.position > b.position;
        }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, LaterCandidate> queue;
    auto enqueue = [&](int64_t position) {
        if (position < 0) return;
        const auto right = next[static_cast<std::size_t>(position)];
        if (right < 0) return;
        const auto found = merge_rank.find(pair_key(values[static_cast<std::size_t>(position)], values[static_cast<std::size_t>(right)]));
        if (found != merge_rank.end())
            queue.push({found->second, position, versions[static_cast<std::size_t>(position)]});
    };
    for (std::size_t i = 0; i + 1 < n; ++i) enqueue(static_cast<int64_t>(i));

    while (!queue.empty()) {
        const auto candidate = queue.top(); queue.pop();
        const auto left_pos = candidate.position;
        if (versions[static_cast<std::size_t>(left_pos)] != candidate.version) continue;
        const auto right_pos = next[static_cast<std::size_t>(left_pos)];
        if (right_pos < 0) continue;
        const auto key = pair_key(values[static_cast<std::size_t>(left_pos)], values[static_cast<std::size_t>(right_pos)]);
        const auto rank = merge_rank.find(key);
        if (rank == merge_rank.end() || rank->second != candidate.rank) continue;

        const auto left_neighbor = previous[static_cast<std::size_t>(left_pos)];
        const auto after = next[static_cast<std::size_t>(right_pos)];
        values[static_cast<std::size_t>(left_pos)] = 256 + candidate.rank;
        next[static_cast<std::size_t>(left_pos)] = after;
        ++versions[static_cast<std::size_t>(left_pos)];
        ++versions[static_cast<std::size_t>(right_pos)];
        previous[static_cast<std::size_t>(right_pos)] = -1;
        next[static_cast<std::size_t>(right_pos)] = -1;
        if (after >= 0) previous[static_cast<std::size_t>(after)] = left_pos;
        if (left_neighbor >= 0) {
            ++versions[static_cast<std::size_t>(left_neighbor)];
            enqueue(left_neighbor);
        }
        enqueue(left_pos);
    }
    std::vector<int64_t> ids; ids.reserve(n);
    int64_t current = 0;
    while (current >= 0) {
        ids.push_back(values[static_cast<std::size_t>(current)]);
        current = next[static_cast<std::size_t>(current)];
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
