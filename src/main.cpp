#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/script.h>

#define VOCAB_MASK_INDEX    103

extern int TokenizersInit(const char* vocab_file);
extern std::vector<float> TokenizersEncode(const char* text);
extern const char* ConvertIdToToken(int idx);

int main(int argc, char *argv[]) {
    int masked_index = 1;
    std::vector<float> input_ids;
    std::vector<float> input_mask;
    const char* text = "The capital of France, [MASK], contains the Eiffel Tower.";
    const char* vocab = "../utils/bert-base-uncased/vocab.txt";
    const char* model = "../utils/bert-base-uncased/traced_bert.pt";

    // generate input id from tokenizer
    TokenizersInit(vocab);
    input_ids = TokenizersEncode(text);
    size_t len = input_ids.size();
    for(size_t i = 0; i < len; i ++) {
        input_mask.push_back(1);
        if((int)input_ids[i] == VOCAB_MASK_INDEX) {
            masked_index = i;
        }
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::from_blob(input_ids.data(), {1, (int)len}).to(torch::kInt64));
    inputs.push_back(torch::from_blob(input_mask.data(), {1, (int)len}).to(torch::kInt64));

    // bert inference
    auto module = torch::jit::load(model);
    module.eval();

    // get output
    auto elms = module.forward(inputs);
    auto dict = elms.toGenericDict();
    auto output = dict.at("logits");
    auto tensor = output.toTensor();
    int64_t k = 10;
    auto result = std::get<1>(tensor[0][masked_index].topk(10, -1, true, true));
    for (int64_t idx = 0; idx < k; idx++) {
        int id = result[idx].item<int>();
        std::string token = ConvertIdToToken(id);
        std::cout << token << "\n";
    }

    return 0;
}

