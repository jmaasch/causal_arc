
# Create a .env file and add your API keys:
#     OPENAI_API_KEY=JM_OA_API_KEY
#     ANTHROPIC_API_KEY=JM_ANT_API_KEY
#     GOOGLE_API_KEY=JM_GOOG_API_KEY

# Examples:
#     python langchain.py -p "How would Judea Pearl define the term 'abstract reasoning'?" --provider openai --model gpt-4o-mini --stream
#     python langchain.py -p "How would Judea Pearl define the term 'abstract reasoning'?" --provider anthropic --model claude-3-5-sonnet-latest --stream
#     python langchain.py -p "How would Judea Pearl define the term 'abstract reasoning'?" --provider gemini --model gemini-1.5-flash --stream

# You can view the models available for each API via the following docs:
# https://ai.google.dev/gemini-api/docs/models
# https://docs.anthropic.com/en/docs/about-claude/models/overview
# https://platform.openai.com/docs/models

import os
import sys
import argparse
from typing import Literal, Optional
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# JM: add these per recommendation here.
# https://stackoverflow.com/questions/78552532/
# what-does-the-error-module-langchain-has-no-attribute-verbose-refer-to
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_aws import ChatBedrock


DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-1.5-flash",
    "bedrock": "us.meta.llama4-scout-17b-instruct-v1:0",
    "xai": "grok-3"
}

def make_llm(
    provider: Literal["openai","anthropic","gemini","xai","bedrock"],
    model: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> Runnable:
    """
    Returns a LangChain chat model runnable.
    If api_key is None, the SDKs will read from env vars automatically.
    """
    model = model or DEFAULTS[provider]

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)
    
    if provider == "bedrock":
        # ChatBedrockConverse uses AWS credentials from environment or IAM role.
        # Pass region explicitly if provided, else rely on AWS_REGION env / default config.
        return ChatBedrock(
            model_id=model,
            region_name="us-east-2",
            temperature=temperature,
        )
    
    if provider == "xai":
        return ChatXAI(model=model, temperature=temperature, api_key=api_key)

    raise ValueError(f"Unknown provider: {provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified LangChain LLM caller")
    parser.add_argument("-p", "--prompt_file", required=True, help="JSON file containing user prompts")
    parser.add_argument("--provider", choices=["openai","anthropic","gemini","xai","bedrock"], default="openai")
    parser.add_argument("--model", help="Model name/alias (defaults per provider)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")

    parser.add_argument("--openai-api-key")
    parser.add_argument("--anthropic-api-key")
    parser.add_argument("--google-api-key")  
    parser.add_argument("--xai-api-key")  
    parser.add_argument("--bedrock-api-key")  
    return parser.parse_args()


def resolve_api_key(provider: str, args: argparse.Namespace) -> Optional[str]:
    if provider == "openai":
        return args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if provider == "anthropic":
        return args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return args.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if provider == "xai":
        return args.xai_api_key or os.getenv("XAI_API_KEY") or os.getenv("XAI_API_KEY")
    return None


def main():
    args = parse_args()
    api_key = resolve_api_key(args.provider, args)

    llm = make_llm(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        api_key=api_key,
    )
    
    # Read prompt json file.
    with open(args.prompt_file) as json_file:
        prompt_dict = json.load(json_file)

    response_dict = dict()
    for task,val in prompt_dict.items():
        
        l1_prompts = val["L1"]
        l3_prompts = val["L3"]
        
        l1_response_dict = dict()
        for rep,prompt in l1_prompts.items():
            msg = llm.invoke(prompt)
            print(getattr(msg, "content", msg))
            l1_response_dict[rep] = msg.text()
            
        l3_response_dict = dict()
        for rep,prompt in l3_prompts.items():
            msg = llm.invoke(prompt)
            print(getattr(msg, "content", msg))
            l3_response_dict[rep] = msg.text()

        response_dict[task] = {"L1": l1_response_dict, 
                               "L3": l3_response_dict}
        
    output_name = args.prompt_file.split(".")[0]
    with open(f"program_synthesis/results_{output_name}_{args.model}.json", "w") as f:
        json.dump(response_dict, f, indent = 4) # indent for readability.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
