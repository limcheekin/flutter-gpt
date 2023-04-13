mkdir bart-lfqa
curl -L https://huggingface.co/vblagoje/bart_lfqa/resolve/main/pytorch_model.bin -o ./bart-lfqa/pytorch_model.bin
curl https://huggingface.co/vblagoje/bart_lfqa/raw/main/config.json -o ./bart-lfqa/config.json
curl https://huggingface.co/vblagoje/bart_lfqa/raw/main/merges.txt -o ./bart-lfqa/merges.txt

curl https://huggingface.co/vblagoje/bart_lfqa/raw/main/tokenizer.json -o ./bart-lfqa/tokenizer.json
curl https://huggingface.co/vblagoje/bart_lfqa/raw/main/tokenizer_config.json -o ./bart-lfqa/tokenizer_config.json
curl https://huggingface.co/vblagoje/bart_lfqa/raw/main/vocab.json -o ./bart-lfqa/vocab.json
