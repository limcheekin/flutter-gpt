mkdir flan-t5-base
curl -L https://huggingface.co/google/flan-t5-base/resolve/main/pytorch_model.bin -o ./flan-t5-base/pytorch_model.bin
curl https://huggingface.co/google/flan-t5-base/raw/main/config.json -o ./flan-t5-base/config.json
curl https://huggingface.co/google/flan-t5-base/raw/main/generation_config.json -o ./flan-t5-base/generation_config.json

curl https://huggingface.co/google/flan-t5-base/raw/main/tokenizer.json -o ./flan-t5-base/tokenizer.json
curl https://huggingface.co/google/flan-t5-base/raw/main/tokenizer_config.json -o ./flan-t5-base/tokenizer_config.json
curl https://huggingface.co/google/flan-t5-base/raw/main/special_tokens_map.json -o ./flan-t5-base/special_tokens_map.json
curl https://huggingface.co/google/flan-t5-base/resolve/main/spiece.model -o ./flan-t5-base/spiece.model
