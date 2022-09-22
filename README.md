# bert_cpp
### prepare
    1. git clone https://github.com/aistream69/bert_cpp.git
    2. sudo apt-get install libicu-dev
    3. download libtorch from https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip
    4. download model from https://huggingface.co/bert-base-uncased/tree/main, 
       and copy config.json/pytorch_model.bin/tokenizer_config.json/tokenizer.json/vocab.txt to bert_cpp/utils/bert-base-uncased
    5. convert model : python3 python/trace.bert.uncased.py
### build
    cd bert_cpp
    mkdir build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/path_to_libtorch ..
    make
### run
    ./sample
    result:
        paris
        lyon
        lille
        toulouse
        marseille
        orleans
        strasbourg
        nice
        cannes
        versailles
    
