#!/bin/bash

# You can view the models available for each API via the following docs:
# https://ai.google.dev/gemini-api/docs/models
# https://docs.anthropic.com/en/docs/about-claude/models/overview
# https://platform.openai.com/docs/models

# Models of interest.
GPT_MODELS=(o4-mini-2025-04-16 gpt-4o-mini-2024-07-18)
CLAUDE_MODELS=(claude-3-5-haiku-20241022 claude-sonnet-4-20250514 claude-opus-4-1-20250805) 
GEMINI_MODELS=(gemini-2.5-flash)
XAI_MODELS=(grok-4)
LLAMA_MODELS=(us.meta.llama4-scout-17b-instruct-v1:0)

# Prompt file.
FILE="../data/cf_reasoning_logical/SCMtcbq_and_xor_n5.json" 
PROGRAM="langchain_cf_reasoning.py" # "langchain_induction.py" "langchain_discovery.py"

# OpenAI models.
# Set --temperature=1 for o4-mini
for model in "${GPT_MODELS[@]}"; do
   echo "Testing $model"
   if [ "$model" = "o4-mini-2025-04-16" ]; then
        echo "Changing temperature."
        python $PROGRAM -p $FILE --provider openai --model "$model" --temperature=1
    elif [ "$model" = "o3-2025-04-16" ]; then
        echo "Changing temperature."
        python $PROGRAM -p $FILE --provider openai --model "$model" --temperature=1
    else
        python $PROGRAM -p $FILE --provider openai --model "$model" 
    fi
done

# Anthropic models.
for model in "${CLAUDE_MODELS[@]}"; do
    echo "Testing $model"
    python $PROGRAM -p $FILE --provider anthropic --model "$model"
done

# Google models.
for model in "${GEMINI_MODELS[@]}"; do
    echo "Testing $model"
    python $PROGRAM -p $FILE --provider gemini --model "$model"
done

# XAI models.
for model in "${XAI_MODELS[@]}"; do
    echo "Testing $model"
    python $PROGRAM -p $FILE --provider xai --model "$model"
done

# Llama models.
for model in "${LLAMA_MODELS[@]}"; do
    echo "Testing $model"
    python $PROGRAM -p $FILE --provider bedrock --model "$model"
done
