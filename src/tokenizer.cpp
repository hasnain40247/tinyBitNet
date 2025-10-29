#include "data/tensor.hpp"
#include "models/transformer.hpp"
#include "criterion/crossEntropy.hpp"
#include "optimizer/optimizer.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <memory>

struct HaikuDataset {
    std::vector<std::vector<int>> inputs;
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;

    // Convert list of words → list of IDs
    std::vector<int> words_to_ids(const std::vector<std::string>& words, int max_seq = 12) const {
        std::vector<int> ids;
        ids.reserve(words.size());
        for (const auto& w : words) {
            auto it = vocab.find(w);
            ids.push_back(it != vocab.end() ? it->second : 0);  // unknown = 0
        }
        if ((int)ids.size() < max_seq)
            ids.resize(max_seq, 0);
        else if ((int)ids.size() > max_seq)
            ids.resize(max_seq);
        return ids;
    }

    // Convert list of IDs → list of words
    std::vector<std::string> ids_to_words(const std::vector<int>& ids) const {
        std::vector<std::string> words;
        words.reserve(ids.size());
        for (int id : ids) {
            if (id == 0) continue;  // skip padding
            auto it = inv_vocab.find(id);
            words.push_back(it != inv_vocab.end() ? it->second : "<unk>");
        }
        return words;
    }
};

HaikuDataset load_haikus(const std::string& filename, int max_seq = 12) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::unordered_map<std::string, int> vocab;
    int next_id = 1;  // 0 reserved for padding
    std::vector<std::vector<int>> tokenized;
    std::string line;

    while (std::getline(file, line, '$')) {
        std::istringstream ss(line);
        std::string token;
        std::vector<int> haiku_tokens;

        while (std::getline(ss, token, '/')) {
            std::istringstream stanza_stream(token);
            std::string word;
            while (stanza_stream >> word) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
                if (word.empty()) continue;
                if (vocab.find(word) == vocab.end()) vocab[word] = next_id++;
                haiku_tokens.push_back(vocab[word]);
            }
        }

        if (haiku_tokens.empty()) continue;

        if ((int)haiku_tokens.size() < max_seq)
            haiku_tokens.resize(max_seq, 0);
        else if ((int)haiku_tokens.size() > max_seq)
            haiku_tokens.resize(max_seq);

        tokenized.push_back(haiku_tokens);
    }

    std::unordered_map<int, std::string> inv_vocab;
    for (auto& kv : vocab) inv_vocab[kv.second] = kv.first;

    return {tokenized, vocab, inv_vocab};
}
