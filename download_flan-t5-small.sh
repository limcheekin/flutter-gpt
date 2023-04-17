mkdir -p google/flan-t5-small
curl -L https://huggingface.co/google/flan-t5-small/resolve/main/pytorch_model.bin -o ./google/flan-t5-small/pytorch_model.bin
curl https://huggingface.co/google/flan-t5-small/raw/main/config.json -o ./google/flan-t5-small/config.json
curl https://huggingface.co/google/flan-t5-small/raw/main/generation_config.json -o ./google/flan-t5-small/generation_config.json

curl https://huggingface.co/google/flan-t5-small/raw/main/tokenizer.json -o ./google/flan-t5-small/tokenizer.json
curl https://huggingface.co/google/flan-t5-small/raw/main/tokenizer_config.json -o ./google/flan-t5-small/tokenizer_config.json
curl https://huggingface.co/google/flan-t5-small/raw/main/special_tokens_map.json -o ./google/flan-t5-small/special_tokens_map.json
curl https://huggingface.co/google/flan-t5-small/resolve/main/spiece.model -o ./google/flan-t5-small/spiece.model
