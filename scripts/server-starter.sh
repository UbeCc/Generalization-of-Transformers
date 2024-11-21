export MODEL=<path to your model>
CUDA_LIST=(0 1 2 3 4 5 6 7)
export PORTS=(18008 18009 18010 18011 18012 18013 18014 18015)
CUDA_LIST_LENGTH=${#CUDA_LIST[@]}
for i in $(seq 0 $((${CUDA_LIST_LENGTH}-1))); do
    export PORT=${PORTS[i]}
    echo Start vllm on port ${PORT}, model is: ${MODEL}
    export CUDA_VISIBLE_DEVICES=${CUDA_LIST[i]}
    nohup python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL} \
        --port ${PORT} \
        --trust-remote-code \
        --gpu-memory-utilization 0.8 \
        --max-model-len 1024 > vllm.log &
done

# test connection (completion)
curl http://localhost:18008/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer this-is-a-local-server" \
    -d '{
      "model": "<path to your model>",
      "stream": false,
      "prompt": "How is the weather today?",
      "temperature": 0.0,
      "max_tokens": 200
    }'

# test connection (chat)
curl http://localhost:18008/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer this-is-a-local-server" \
    -d '{
      "model": "<path to your model>",
      "stream": false,
      "messages": [{"role": "user", "content": "Who are you?"}],
      "temperature": 0.0,
      "max_tokens": 200
    }'