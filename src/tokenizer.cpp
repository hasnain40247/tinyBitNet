// #include <iostream>
// #include <string>
// #include <vector>
// #include <unordered_map>
// #include <fstream>
// #include <sstream>
// #include <algorithm>
// #include <cctype>

// class WordTokenizer {
// private:
//     // the crux of it all is to have a word to id and id to word for encoding and decoding
//     std::unordered_map<std::string, int> word_to_id;
//     std::unordered_map<int, std::string> id_to_word;
//     int next_id = 0;
    
//     // spcl tokens ig
//     int pad_token_id;
//     int unk_token_id;  
//     int start_token_id;
//     int end_token_id;

// public:
//     WordTokenizer() {
        
//         add_special_token("<PAD>");  
//         add_special_token("<UNK>");   
//         add_special_token("<START>"); 
//         add_special_token("<END>");  
        
//         pad_token_id = 0;
//         unk_token_id = 1;
//         start_token_id = 2;
//         end_token_id = 3;
//     }
    
// private:
//     void add_token(const std::string& token) {
//         word_to_id[token] = next_id;
//         id_to_word[next_id] = token;
//         next_id++;
//     }
    
//     // just lowercase it
//     std::string normalize_word(const std::string& word) {
//         std::string cleaned;
        
//         for (char c : word) {
//             if (std::isalnum(c)) {
//                 cleaned += std::tolower(c);
//             }
//         }
        
//         return cleaned;
//     }
    
//     // takes the text and 
//     std::vector<std::string> split_into_words(const std::string& text) {
//         std::vector<std::string> words;
//         std::string current_word;
        
//         for (size_t i = 0; i < text.length(); i++) {
//             char c = text[i];
            
//             if (std::isspace(c)) {
//              =
//                 if (!current_word.empty()) {
//                     std::string normalized = normalize_word(current_word);
//                     if (!normalized.empty()) {
//                         words.push_back(normalized);
//                     }
//                     current_word.clear();
//                 }
//             }
//             else if (std::ispunct(c)) {
// =
//                 if (!current_word.empty()) {
//                     std::string normalized = normalize_word(current_word);
//                     if (!normalized.empty()) {
//                         words.push_back(normalized);
//                     }
//                     current_word.clear();
//                 }
//               =
//             }
//             else {
           
//                 current_word += c;
//             }
//         }
        
//         if (!current_word.empty()) {
//             std::string normalized = normalize_word(current_word);
//             if (!normalized.empty()) {
//                 words.push_back(normalized);
//             }
//         }
        
//         return words;
//     }
    
// public:
   
//     void build_vocabulary(const std::vector<std::string>& texts) {
//         std::unordered_map<std::string, int> word_counts;
        
//         // Count all words
//         for (const auto& text : texts) {
//             auto words = split_into_words(text);
//             for (const auto& word : words) {
//                 word_counts[word]++;
//             }
//         }
        
//         std::cout << "Found " << word_counts.size() << " unique words" << std::endl;
        
//         // Add words to vocabulary (sorted by frequency)
//         std::vector<std::pair<std::string, int>> sorted_words(word_counts.begin(), word_counts.end());
//         std::sort(sorted_words.begin(), sorted_words.end(), 
//                   [](const auto& a, const auto& b) { return a.second > b.second; });
        
//         // Add words to vocabulary
//         for (const auto& pair : sorted_words) {
//             if (word_to_id.find(pair.first) == word_to_id.end()) {
//                 word_to_id[pair.first] = next_id;
//                 id_to_word[next_id] = pair.first;
//                 next_id++;
//             }
//         }
        
//         std::cout << "Built vocabulary of size: " << next_id << std::endl;
        
//         // Print most common words
//         std::cout << "Most common words:" << std::endl;
//         for (int i = 0; i < std::min(10, (int)sorted_words.size()); i++) {
//             std::cout << "  " << sorted_words[i].first << " (count: " << sorted_words[i].second << ")" << std::endl;
//         }
//     }
    
//     // Tokenize text to token IDs
//     std::vector<int> encode(const std::string& text) {
//         std::vector<int> token_ids;
        
//         // Add start token
//         token_ids.push_back(start_token_id);
        
//         // Tokenize words
//         auto words = split_into_words(text);
//         for (const auto& word : words) {
//             auto it = word_to_id.find(word);
//             if (it != word_to_id.end()) {
//                 token_ids.push_back(it->second);
//             } else {
//                 token_ids.push_back(unk_token_id);  // Unknown word
//             }
//         }
        
//         // Add end token
//         token_ids.push_back(end_token_id);
        
//         return token_ids;
//     }
    
//     std::string decode(const std::vector<int>& token_ids) {
//         std::vector<std::string> words;
        
//         for (int id : token_ids) {
//             auto it = id_to_word.find(id);
//             if (it != id_to_word.end()) {
//                 const std::string& token = it->second;
//                 // Skip special tokens except for debugging
//                 if (token != "<PAD>" && token != "<START>" && token != "<END>") {
//                     words.push_back(token);
//                 }
//             }
//         }
        
//         // Join words with spaces
//         std::string result;
//         for (size_t i = 0; i < words.size(); i++) {
//             if (i > 0) result += " ";
//             result += words[i];
//         }
        
//         return result;
//     }
    
//     // Utility functions
//     int vocab_size() const { return next_id; }
//     int get_pad_token_id() const { return pad_token_id; }
//     int get_unk_token_id() const { return unk_token_id; }
//     int get_start_token_id() const { return start_token_id; }
//     int get_end_token_id() const { return end_token_id; }
    
//     // Get word from ID (for debugging)
//     std::string id_to_word_debug(int id) const {
//         auto it = id_to_word.find(id);
//         if (it != id_to_word.end()) {
//             return it->second;
//         }
//         return "<INVALID>";
//     }
    
//     // Print vocabulary (for debugging)
//     void print_vocabulary() const {
//         std::cout << "\n=== Vocabulary ===" << std::endl;
//         for (int i = 0; i < std::min(20, next_id); i++) {
//             std::cout << i << ": " << id_to_word_debug(i) << std::endl;
//         }
//         if (next_id > 20) {
//             std::cout << "... (and " << (next_id - 20) << " more)" << std::endl;
//         }
//         std::cout << "==================" << std::endl;
//     }
// };

// // Main function to load and process JSONL haiku data
// int main() {
//     HaikuDataset dataset;
    
//     // Load from JSONL file
//     std::cout << "Loading haiku dataset from JSONL..." << std::endl;
//     dataset.load_from_jsonl("haikus.jsonl");  // Make sure this file exists!
    
//     // Build vocabulary
//     std::cout << "\nBuilding vocabulary..." << std::endl;
//     dataset.prepare();
    
//     // Print vocabulary for inspection
//     dataset.get_tokenizer().print_vocabulary();
    
//     // Test tokenization on a few examples
//     auto& tokenizer = dataset.get_tokenizer();
//     auto all_tokenized = dataset.get_all_tokenized();
    
//     std::cout << "\n=== Tokenization Examples ===" << std::endl;
//     for (int i = 0; i < std::min(3, (int)all_tokenized.size()); i++) {
//         auto& tokens = all_tokenized[i];
//         std::string decoded = tokenizer.decode(tokens);
        
//         std::cout << "\nHaiku " << (i+1) << ":" << std::endl;
//         std::cout << "Tokens: ";
//         for (int token : tokens) {
//             std::cout << token << "(" << tokenizer.id_to_word_debug(token) << ") ";
//         }
//         std::cout << std::endl;
//         std::cout << "Decoded: " << decoded << std::endl;
//     }
    
//     std::cout << "\n=== Dataset Statistics ===" << std::endl;
//     std::cout << "Total haikus: " << dataset.size() << std::endl;
//     std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    
//     // Calculate average sequence length
//     double total_length = 0;
//     for (const auto& tokens : all_tokenized) {
//         total_length += tokens.size();
//     }
//     double avg_length = total_length / all_tokenized.size();
//     std::cout << "Average sequence length: " << avg_length << " tokens" << std::endl;
    
//     return 0;
// }

// class HaikuDataset {
// private:
//     std::vector<std::string> haiku_texts;
//     WordTokenizer tokenizer;
    
//     std::string extract_text_from_jsonl(const std::string& line) {
//         size_t start = line.find("\"text\": \"");
//         if (start == std::string::npos) return "";
        
//         start += 9;
//         size_t end = line.find("\"}", start);
//         if (end == std::string::npos) return "";
        
//         std::string text = line.substr(start, end - start);
        

//         size_t pos = 0;
//         while ((pos = text.find(" / ", pos)) != std::string::npos) {
//             text.replace(pos, 3, " ");
//             pos += 1;
//         }
        
//         return text;
//     }
    
// public:
//     void load_from_jsonl(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
        
//         while (std::getline(file, line)) {
//             std::string haiku = extract_text_from_jsonl(line);
//             if (!haiku.empty()) {
//                 haiku_texts.push_back(haiku);
//             }
//         }
        
//         std::cout << "Loaded " << haiku_texts.size() << " haikus" << std::endl;
//     }
    
//     void prepare() {
//         tokenizer.build_vocabulary(haiku_texts);
//     }
    
//     std::vector<std::vector<int>> get_all_tokenized() {
//         std::vector<std::vector<int>> all_tokens;
//         for (const auto& text : haiku_texts) {
//             all_tokens.push_back(tokenizer.encode(text));
//         }
//         return all_tokens;
//     }
    
//     WordTokenizer& get_tokenizer() { return tokenizer; }
//     size_t size() const { return haiku_texts.size(); }
// };