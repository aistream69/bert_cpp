#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <memory>
#include "tokenizers/bert/bert_tokenizer.h"
#include "tokenizers/fundamental/fundamental_tokenizer.h"

std::unique_ptr<tokenizers::BertTokenizer> bert_tokenizer_ = nullptr;
int TokenizersInit(const char* vocab_file) {
    tokenizers::BertTokenizer::Options bert_options;
    bert_options.vocab_file = vocab_file;
    bert_tokenizer_ = tokenizers::BertTokenizer::CreateTokenizer(bert_options);
    return 0;
}

std::vector<float> TokenizersEncode(const char* text) {
    std::vector<float> input_ids;
    std::string _text = text;
    auto output = bert_tokenizer_->Encode(&_text);
    for(size_t i = 0; i < output.input_ids.size(); i++) {
        if(output.input_ids[i] == 0) {
            break;
        }
        input_ids.push_back(output.input_ids[i]);
    }
    return input_ids;
}

const char* ConvertIdToToken(int idx) {
    std::string converted;
    auto us = bert_tokenizer_->ConvertIdToToken(idx);
    us.toUTF8String(converted);
    return converted.c_str();
}

