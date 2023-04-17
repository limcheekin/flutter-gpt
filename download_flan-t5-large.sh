mkdir -p google/flan-t5-large
curl -L https://huggingface.co/google/flan-t5-large/resolve/main/pytorch_model.bin -o ./google/flan-t5-large/pytorch_model.bin
curl https://huggingface.co/google/flan-t5-large/raw/main/config.json -o ./google/flan-t5-large/config.json
curl https://huggingface.co/google/flan-t5-large/raw/main/generation_config.json -o ./google/flan-t5-large/generation_config.json

curl https://huggingface.co/google/flan-t5-large/raw/main/tokenizer.json -o ./google/flan-t5-large/tokenizer.json
curl https://huggingface.co/google/flan-t5-large/raw/main/tokenizer_config.json -o ./google/flan-t5-large/tokenizer_config.json
curl https://huggingface.co/google/flan-t5-large/raw/main/special_tokens_map.json -o ./google/flan-t5-large/special_tokens_map.json
curl https://huggingface.co/google/flan-t5-large/resolve/main/spiece.model -o ./google/flan-t5-large/spiece.model
