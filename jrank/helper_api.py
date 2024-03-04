"""API LLM answer generation helper functions."""
import os
import time

import openai
from openai import OpenAI
import anthropic
from fastchat.serve.api_provider import init_palm_chat

from common import API_ERROR_OUTPUT, API_MAX_RETRY, API_RETRY_SLEEP


def chat_completion_openai(
    model, conv, temperature: float = 0.0, max_tokens: int = 1024
):
    """Chat completion using OpenAI API."""
    output = API_ERROR_OUTPUT
    print("MAX TOKEN: ", max_tokens)

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            # print(messages)
            client = OpenAI()
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_completion.choices[0].message.content
            break
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            time.sleep(API_RETRY_SLEEP)
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            print(e.__cause__)
            time.sleep(API_RETRY_SLEEP)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response.json())
            time.sleep(API_RETRY_SLEEP)

    return output


def chat_completion_anthropic(
    model, conv, temperature: float = 0.0, max_tokens: int = 2048
):
    """Chat completion using Anthropic API."""
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_palm(
    chat_state, model, conv, temperature: float = 0.0, max_tokens: int = 2048
):
    """Chat completion using PALM API."""

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output
