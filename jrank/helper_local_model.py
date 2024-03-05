"""Helper functions for local model answer generation."""
import json

from fastchat.conversation import Conversation, SeparatorStyle


def get_conv_from_template_path(template_path):
    """Helper that generate a fastchat conversation from a template file"""
    with open(template_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Convert sep_style from string to SeparatorStyle enum
    if "sep_style" in config:
        config["sep_style"] = SeparatorStyle[config["sep_style"]]

    # Start a conversation
    if "messages" not in config:
        config["messages"] = []

    return Conversation(**config)
