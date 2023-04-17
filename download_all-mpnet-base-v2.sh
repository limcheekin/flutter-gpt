mkdir -p sentence-transformers/all-mpnet-base-v2/1_Pooling
curl -L https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/pytorch_model.bin -o ./sentence-transformers/all-mpnet-base-v2/pytorch_model.bin
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/sentence_bert_config.json -o ./sentence-transformers/all-mpnet-base-v2/sentence_bert_config.json
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/special_tokens_map.json -o ./sentence-transformers/all-mpnet-base-v2/special_tokens_map.json

curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/tokenizer.json -o ./sentence-transformers/all-mpnet-base-v2/tokenizer.json
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/tokenizer_config.json -o ./sentence-transformers/all-mpnet-base-v2/tokenizer_config.json
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/train_script.py -o ./sentence-transformers/all-mpnet-base-v2/train_script.py
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/vocab.txt -o ./sentence-transformers/all-mpnet-base-v2/vocab.txt
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/modules.json -o ./sentence-transformers/all-mpnet-base-v2/modules.json
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/config.json -o ./sentence-transformers/all-mpnet-base-v2/config.json
curl https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/1_Pooling/config.json -o ./sentence-transformers/all-mpnet-base-v2/1_Pooling/config.json