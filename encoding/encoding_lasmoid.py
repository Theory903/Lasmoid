"""
Lasmoid — encoding_lasmoid.py
=============================
A self-contained implementation for encoding/decoding Lasmoid chat messages
with tool calling, cognitive reasoning modes, and structured agentic operating system blocks.
"""

from typing import Any, Dict, List, Union, Optional, Tuple
import copy
from dataclasses import dataclass
from html import escape as html_escape, unescape as html_unescape
import json
import re

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"
DEVELOPER_SP_TOKEN = "<｜Developer｜>"

MULTIMODAL_SP_TOKENS = {
    "image": "<｜image｜>",
    "audio": "<｜audio｜>",
    "video": "<｜video｜>",
    "file": "<｜file｜>",
    "embedding": "<｜embedding｜>",
}

FRONTIER_CHANNELS = ("analysis", "commentary", "final")

# Task special tokens for internal classification tasks
LASMOID_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(LASMOID_TASK_SP_TOKENS.keys())


@dataclass(frozen=True)
class TokenizerProfile:
    """Tokenizer contract expected by Lasmoid prompt encoding."""
    vocab_size: int = 128000
    model_max_length: int = 1048576
    bos_token: str = bos_token
    eos_token: str = eos_token
    pad_token: str = eos_token
    chat_special_tokens: Tuple[str, ...] = (
        USER_SP_TOKEN,
        ASSISTANT_SP_TOKEN,
        DEVELOPER_SP_TOKEN,
        LATEST_REMINDER_SP_TOKEN,
        thinking_start_token,
        thinking_end_token,
        dsml_token,
    )
    multimodal_special_tokens: Tuple[str, ...] = tuple(MULTIMODAL_SP_TOKENS.values())

    @property
    def required_special_tokens(self) -> Tuple[str, ...]:
        return (
            self.bos_token,
            self.eos_token,
            self.pad_token,
            *self.chat_special_tokens,
            *self.multimodal_special_tokens,
            *LASMOID_TASK_SP_TOKENS.values(),
        )


LASMOID_TOKENIZER_PROFILE = TokenizerProfile()

# ============================================================
# Templates & Policies
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

# Structured response format instructions
response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
tool_call_v2_template: str = (
    "<{dsml_token}invoke name=\"{name}\""
    "{confidence_attr}"
    "{reason_attr}"
    "{expected_info_attr}>\n"
    "{arguments}\n"
    "</{dsml_token}invoke>"
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"
parallel_tool_calls_block_name: str = "parallel_tool_calls"

tool_output_template: str = (
    "<tool_result>{content}</tool_result>"
)

# Backwards compatible reasoning effort prompt
REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)

# Updated Frontier cognitive reasoning policy
REASONING_POLICY = """Think privately and persistently.
Use reasoning only to:
1. Identify the actual task and establish an exact algorithmic plan.
2. Detect missing information across the global context.
3. Track your own prior failed tool calls to prevent answer thrashing.
4. Verify conclusions deterministically.
Do not discard prior hypotheses; instead, formally invalidate them in your <think> block before pivoting.
"""

# V1 tool template for backwards compatibility tests
TOOLS_TEMPLATE_V1 = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# V2 tool template with parallel tool execution docs
TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" or "<{dsml_token}parallel_tool_calls>" block like the following:

<{dsml_token}parallel_tool_calls>
<{dsml_token}invoke name="$TOOL_NAME" confidence="0.95" reason="missing_info" expected_information="latest_specs">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
</{dsml_token}parallel_tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode/reasoning_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ============================================================
# Utility Functions
# ============================================================

def to_json(value: Any) -> str:
    """Serialize a value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def escape_dsml_text(value: Any) -> str:
    """Escape raw text embedded inside DSML/XML-like tags."""
    return html_escape(str(value), quote=False)


def unescape_dsml_text(value: str) -> str:
    """Decode text escaped by escape_dsml_text."""
    return html_unescape(value)


def escape_dsml_attr(value: Any) -> str:
    """Escape attribute values embedded in DSML tags."""
    return html_escape(str(value), quote=True)


def normalize_text_content(content: Any) -> str:
    """Render string, OpenAI, Gemini, and Gemma-style content into Lasmoid text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(render_content_block(block) for block in content)
    return str(content)


def render_content_block(block: Any) -> str:
    """Render a multimodal or structured content block into stable text."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return str(block)

    block_type = block.get("type", "text")
    if block_type in {"text", "input_text", "output_text"}:
        return str(block.get("text", ""))
    if block_type in {"image", "input_image", "image_url"}:
        payload = block.get("image_url", block.get("url", block.get("source", "")))
        return f'{MULTIMODAL_SP_TOKENS["image"]}{to_json(payload)}'
    if block_type in {"audio", "input_audio"}:
        payload = {k: v for k, v in block.items() if k != "type"}
        return f'{MULTIMODAL_SP_TOKENS["audio"]}{to_json(payload)}'
    if block_type in {"video", "input_video"}:
        payload = {k: v for k, v in block.items() if k != "type"}
        return f'{MULTIMODAL_SP_TOKENS["video"]}{to_json(payload)}'
    if block_type in {"file", "input_file"}:
        payload = {k: v for k, v in block.items() if k != "type"}
        return f'{MULTIMODAL_SP_TOKENS["file"]}{to_json(payload)}'
    if block_type in {"embedding", "semantic_embedding"}:
        payload = {k: v for k, v in block.items() if k != "type"}
        return f'{MULTIMODAL_SP_TOKENS["embedding"]}{to_json(payload)}'
    if block_type == "tool_result":
        return render_tool_result_block(block)
    return f"[Unsupported {block_type}]"


def render_tool_result_block(block: Dict[str, Any]) -> str:
    """Render tool results with stable JSON envelopes when metadata exists."""
    tool_content = block.get("content", "")

    if any(k in block for k in ["tool", "status", "confidence", "results", "error"]):
        tool_data = {
            "tool": block.get("tool", ""),
            "tool_use_id": block.get("tool_use_id", ""),
            "status": block.get("status", "success"),
            "confidence": block.get("confidence", 1.0),
            "results": block.get("results", []),
        }
        if "error" in block:
            tool_data["error"] = block["error"]
        if tool_content not in ("", None):
            tool_data["content"] = tool_content
        return tool_output_template.format(content=escape_dsml_text(to_json(tool_data)))

    if isinstance(tool_content, list):
        tool_content = "\n\n".join(render_content_block(b) for b in tool_content)
    return tool_output_template.format(content=escape_dsml_text(tool_content))


def validate_tokenizer_config(tokenizer_config: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """Check a tokenizer config against the Lasmoid chat/agentic contract."""
    profile = LASMOID_TOKENIZER_PROFILE
    missing = []
    mismatched = []

    def _token_content(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            return value.get("content")
        if isinstance(value, str):
            return value
        return None

    for key, expected in {
        "bos_token": profile.bos_token,
        "eos_token": profile.eos_token,
        "pad_token": profile.pad_token,
    }.items():
        actual = _token_content(tokenizer_config.get(key))
        if actual is None:
            missing.append(key)
        elif actual != expected:
            mismatched.append({"field": key, "expected": expected, "actual": actual})

    model_max_length = tokenizer_config.get("model_max_length")
    if model_max_length is not None and int(model_max_length) < profile.model_max_length:
        mismatched.append({
            "field": "model_max_length",
            "expected": f">={profile.model_max_length}",
            "actual": model_max_length,
        })

    result = {
        "ok": not missing and not mismatched,
        "missing": missing,
        "mismatched": mismatched,
        "required_special_tokens": list(dict.fromkeys(profile.required_special_tokens)),
    }
    if strict and not result["ok"]:
        raise ValueError(f"Tokenizer config is incompatible with Lasmoid encoding: {result}")
    return result


def _format_agentic_block(name: str, value: Any, delimiter: str = ":") -> str:
    """Render deterministic internal state blocks without losing nested structure."""
    if isinstance(value, dict):
        simple = all(not isinstance(v, (dict, list, tuple)) for v in value.values())
        if simple:
            sep = delimiter if delimiter == "=" else f"{delimiter} "
            body = "\n".join(f"{k}{sep}{v}" for k, v in value.items())
        else:
            body = to_json(value)
    elif isinstance(value, list):
        body = to_json(value)
    else:
        body = str(value)
    return f"<{name}>\n{body}\n</{name}>"


def tools_from_openai_format(tools):
    """Extract function definitions from OpenAI-format tool list."""
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """Convert OpenAI-format tool calls to internal format with attributes."""
    res = []
    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        item = {
            "name": fn.get("name"),
            "arguments": fn.get("arguments"),
        }
        for attr in ["confidence", "reason", "expected_information"]:
            if attr in tool_call:
                item[attr] = tool_call[attr]
        res.append(item)
    return res


def tool_calls_to_openai_format(tool_calls):
    """Convert internal tool calls to OpenAI format with attributes."""
    res = []
    for tc in tool_calls:
        item = {
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"],
            }
        }
        for attr in ["confidence", "reason", "expected_information"]:
            if attr in tc:
                item[attr] = tc[attr]
        res.append(item)
    return res


def encode_arguments_to_dsml(tool_call: Dict[str, str]) -> str:
    """Encode tool call arguments into DSML parameter format."""
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    p_dsml_strs = []

    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception:
        arguments = {"arguments": tool_call["arguments"]}

    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=escape_dsml_attr(k),
            is_str="true" if isinstance(v, str) else "false",
            value=escape_dsml_text(v if isinstance(v, str) else to_json(v)),
        )
        p_dsml_strs.append(p_dsml_str)

    return "\n".join(p_dsml_strs)


def decode_dsml_to_arguments(tool_name: str, tool_args: Dict[str, Tuple[str, str]]) -> Dict[str, str]:
    """Decode DSML parameters back to a tool call dict."""
    def _decode_value(key: str, value: str, string: str):
        key = unescape_dsml_text(key)
        value = unescape_dsml_text(value)
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"

    tool_args_json = "{" + ", ".join([_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}"
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]], is_v2: bool = True) -> str:
    """Render tool schemas into the system prompt format."""
    tools_json = [to_json(t) for t in tools]
    template = TOOLS_TEMPLATE if is_v2 else TOOLS_TEMPLATE_V1

    return template.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Find the index of the last user/developer message."""
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# Message Rendering
# ============================================================

def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    reasoning_mode: str = "normal",
    tool_mode: str = "optional",
    drop_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    thinking_mode: Optional[str] = None
) -> str:
    """Render a single message at the given index into its encoded string form."""
    assert 0 <= index < len(messages)

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0:
        if reasoning_effort == 'max':
            if thinking_mode == "thinking":
                prompt += REASONING_EFFORT_MAX
            else:
                prompt += REASONING_POLICY

    is_v2 = (thinking_mode is None)

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools and tool_mode != "disabled":
            prompt += "\n\n" + render_tools(tools, is_v2=is_v2)
        if response_format:
            prompt += "\n\n" + response_format_template.format(schema=to_json(response_format))

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += normalize_text_content(content)

        if tools and tool_mode != "disabled":
            content_developer += "\n\n" + render_tools(tools, is_v2=is_v2)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(schema=to_json(response_format))

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN

        content_blocks = msg.get("content_blocks")
        if content_blocks:
            prompt += "\n\n".join(render_content_block(block) for block in content_blocks)
        else:
            prompt += normalize_text_content(content)

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(content=content)

    elif role == "tool":
        raise NotImplementedError("Lasmoid merges tool messages into user; please preprocess with merge_tool_messages()")

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls and tool_mode != "disabled":
            tc_list = []
            for tc in tool_calls:
                confidence_attr = f' confidence="{escape_dsml_attr(tc.get("confidence"))}"' if tc.get("confidence") is not None else ""
                reason_attr = f' reason="{escape_dsml_attr(tc.get("reason"))}"' if tc.get("reason") is not None else ""
                expected_info_attr = f' expected_information="{escape_dsml_attr(tc.get("expected_information"))}"' if tc.get("expected_information") is not None else ""
                
                tc_list.append(
                    tool_call_v2_template.format(
                        dsml_token=dsml_token,
                        name=escape_dsml_attr(tc.get("name")),
                        confidence_attr=confidence_attr,
                        reason_attr=reason_attr,
                        expected_info_attr=expected_info_attr,
                        arguments=encode_arguments_to_dsml(tc)
                    )
                )
            
            is_parallel = msg.get("parallel", False) or len(tool_calls) > 1
            tc_block_name = parallel_tool_calls_block_name if is_parallel else tool_calls_block_name
            
            tc_content += '\n\n' + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tc_block_name,
            )

        summary_content = normalize_text_content(content)
        rc = reasoning_content or ""

        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if reasoning_mode != "fast" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                # Format structured agentic blocks
                agentic_parts = []
                
                # 1. Working Memory
                if msg.get("working_memory"):
                    agentic_parts.append(_format_agentic_block("working_memory", msg.get("working_memory"), ":"))
                
                # 2. First-principles Plan
                if msg.get("plan"):
                    agentic_parts.append(_format_agentic_block("plan", msg.get("plan"), ":"))
                
                # 3. Evidence Check
                if msg.get("evidence_check"):
                    agentic_parts.append(_format_agentic_block("evidence_check", msg.get("evidence_check"), ":"))
                
                # 4. Tool Cost/Decision
                if msg.get("tool_decision"):
                    agentic_parts.append(_format_agentic_block("tool_decision", msg.get("tool_decision"), "="))

                # 5. Research Mode
                if msg.get("research"):
                    agentic_parts.append(_format_agentic_block("research", msg.get("research"), ":"))

                # 6. Verification
                if msg.get("verification"):
                    agentic_parts.append(_format_agentic_block("verification", msg.get("verification"), "="))

                # 7. Critique
                if msg.get("critique"):
                    agentic_parts.append(_format_agentic_block("critique", msg.get("critique"), ":"))

                # 8. Self-Review
                if msg.get("self_review"):
                    agentic_parts.append(_format_agentic_block("self_review", msg.get("self_review"), ":"))

                combined_reasoning = "\n\n".join(agentic_parts)
                if rc:
                    if combined_reasoning:
                        combined_reasoning += "\n\n"
                    combined_reasoning += rc

                thinking_part = thinking_template.format(reasoning_content=combined_reasoning) + thinking_end_token
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        assert task in VALID_TASKS, f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        task_sp_token = LASMOID_TASK_SP_TOKENS[task]

        if task != "action":
            prompt += task_sp_token
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += thinking_end_token if reasoning_mode == "fast" else thinking_start_token
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"]:
        prompt += ASSISTANT_SP_TOKEN
        if reasoning_mode != "fast":
            if not drop_thinking:
                prompt += thinking_start_token
            elif index >= last_user_idx:
                prompt += thinking_start_token
            else:
                prompt += thinking_end_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Preprocessing
# ============================================================

def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge tool messages into the preceding user message using content_blocks format."""
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # propagate structured metadata if present
            for k in ["tool", "status", "confidence", "results"]:
                if k in msg:
                    tool_block[k] = msg[k]
                    
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({
                    "role": "user",
                    "content_blocks": [tool_block],
                })
        elif role == "user":
            raw_content = msg.get("content", "")
            if isinstance(raw_content, list):
                content_blocks = raw_content
                rendered_content = normalize_text_content(raw_content)
            else:
                rendered_content = normalize_text_content(raw_content)
                content_blocks = [{"type": "text", "text": rendered_content}]

            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1] and merged[-1].get("task") is None:
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                new_msg = {
                    "role": "user",
                    "content": rendered_content,
                    "content_blocks": content_blocks,
                }
                for key in ("task", "wo_eos", "mask", "parallel"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort tool_result blocks within user messages by the order of tool_calls."""
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0)
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Main Encoding Function
# ============================================================

def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: Optional[str] = None,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Optional[str] = None,
    reasoning_mode: str = "normal",
    tool_mode: str = "optional",
) -> str:
    """Encode a list of messages into the Lasmoid prompt format."""
    context = context if context else []

    # Map deprecated thinking_mode for backwards compatibility
    if thinking_mode is not None:
        if thinking_mode == "thinking":
            reasoning_mode = "normal"
        elif thinking_mode == "chat":
            reasoning_mode = "fast"

    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if reasoning_mode != "fast" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            reasoning_mode=reasoning_mode,
            tool_mode=tool_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
        )

    return prompt


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop reasoning_content and non-essential messages before the last user message."""
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)

    return result


# ============================================================
# Parsing (Decoding model output)
# ============================================================

def _read_until_stop(index: int, text: str, stop: List[str]) -> Tuple[int, str, Optional[str]]:
    """Read text from index until one of the stop strings is found."""
    min_pos = len(text)
    matched_stop = None

    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s

    if matched_stop:
        content = text[index:min_pos]
        return min_pos + len(matched_stop), content, matched_stop
    else:
        content = text[index:]
        return len(text), content, None


def parse_tool_calls(index: int, text: str) -> Tuple[int, Optional[str], List[Dict[str, str]]]:
    """Parse DSML tool calls from text starting at the given index."""
    tool_calls: List[Dict[str, Any]] = []
    stop_token = None
    
    end_tokens = [f"</{dsml_token}{tool_calls_block_name}>", f"</{dsml_token}{parallel_tool_calls_block_name}>"]

    while index < len(text):
        index, _, stop_token = _read_until_stop(index, text, [f"<{dsml_token}invoke"] + end_tokens)
        if _ != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{_}'")

        if stop_token in end_tokens:
            break

        if stop_token is None:
            raise ValueError("Missing special token in tool calls")

        index, tool_name_content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])

        # Parse name and attributes like confidence, reason, expected_information
        attribs = {k: unescape_dsml_text(v) for k, v in re.findall(r'(\w+)="(.*?)"', tool_name_content)}
        tool_name = attribs.get("name")
        if not tool_name:
            raise ValueError(f"Tool name format error: missing 'name' in '{tool_name_content}'")

        tool_args: Dict[str, Tuple[str, str]] = {}
        while stop_token == f"<{dsml_token}parameter":
            index, param_content, stop_token = _read_until_stop(index, text, [f"/{dsml_token}parameter"])

            param_kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$', param_content, flags=re.DOTALL)
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]
            param_name = unescape_dsml_text(param_name)

            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)

            index, content, stop_token = _read_until_stop(index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])
            if content != ">\n":
                raise ValueError(f"Parameter format error: expected '>\\n' but got '{content}'")

        tool_call = decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args)
        # Add extra attributes
        for attr in ["confidence", "reason", "expected_information"]:
            if attr in attribs:
                tool_call[attr] = attribs[attr]
        tool_calls.append(tool_call)

    return index, stop_token, tool_calls


def extract_tag_content(text: str, tag_name: str) -> Optional[str]:
    """Extract content between <tag> and </tag>."""
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    return text[start_idx + len(start_tag):end_idx].strip()


def parse_key_value_pairs(content: str, delimiter: str = ":") -> Dict[str, str]:
    """Parse lines of key-value pairs separated by either : or =."""
    stripped = content.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            pass
    res = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if delimiter in line:
            k, v = line.split(delimiter, 1)
            res[k.strip()] = v.strip()
    return res


def parse_message_from_completion_text(text: str, thinking_mode: str) -> Dict[str, Any]:
    """Parse a model completion text into a structured assistant message."""
    summary_content, reasoning_content, tool_calls = "", "", []
    index, stop_token = 0, None
    
    tool_calls_start_token = f"\n\n<{dsml_token}{tool_calls_block_name}"
    parallel_start_token = f"\n\n<{dsml_token}{parallel_tool_calls_block_name}"

    is_thinking = thinking_mode == "thinking"
    is_tool_calling = False

    if is_thinking:
        index, content_delta, stop_token = _read_until_stop(index, text, [thinking_end_token, tool_calls_start_token, parallel_start_token])
        reasoning_content = content_delta
        assert stop_token == thinking_end_token, "Invalid thinking format: missing </think>"

    index, content_delta, stop_token = _read_until_stop(index, text, [eos_token, tool_calls_start_token, parallel_start_token])
    summary_content = content_delta
    if stop_token in [tool_calls_start_token, parallel_start_token]:
        is_tool_calling = True
    else:
        assert stop_token == eos_token, "Invalid format: missing EOS token"

    if is_tool_calling:
        index, stop_token, tool_calls = parse_tool_calls(index, text)

        index, tool_ends_text, stop_token = _read_until_stop(index, text, [eos_token])
        assert not tool_ends_text, "Unexpected content after tool calls"

    assert len(text) == index and stop_token in [eos_token, None], "Unexpected content at end"

    for sp_token in [bos_token, eos_token, thinking_start_token, thinking_end_token, dsml_token]:
        assert sp_token not in summary_content, f"Unexpected special token '{sp_token}' in content"

    # Extract structured blocks from reasoning_content
    plan_content = extract_tag_content(reasoning_content, "plan")
    ev_content = extract_tag_content(reasoning_content, "evidence_check")
    wm_content = extract_tag_content(reasoning_content, "working_memory")
    td_content = extract_tag_content(reasoning_content, "tool_decision")
    ver_content = extract_tag_content(reasoning_content, "verification")
    crit_content = extract_tag_content(reasoning_content, "critique")
    res_content = extract_tag_content(reasoning_content, "research")
    sr_content = extract_tag_content(reasoning_content, "self_review")

    # Clean the reasoning content by removing structured tags
    clean_rc = reasoning_content
    for tag in ["plan", "evidence_check", "working_memory", "tool_decision", "verification", "critique", "research", "self_review"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        while True:
            s_idx = clean_rc.find(start_tag)
            if s_idx == -1:
                break
            e_idx = clean_rc.find(end_tag, s_idx)
            if e_idx == -1:
                break
            clean_rc = clean_rc[:s_idx] + clean_rc[e_idx + len(end_tag):]
    clean_rc = clean_rc.strip()

    res_dict = {
        "role": "assistant",
        "content": summary_content,
        "reasoning_content": clean_rc,
        "tool_calls": tool_calls_to_openai_format(tool_calls)
    }

    if plan_content is not None:
        res_dict["plan"] = parse_key_value_pairs(plan_content, ":")
    if ev_content is not None:
        res_dict["evidence_check"] = parse_key_value_pairs(ev_content, ":")
    if wm_content is not None:
        res_dict["working_memory"] = parse_key_value_pairs(wm_content, ":")
    if td_content is not None:
        res_dict["tool_decision"] = parse_key_value_pairs(td_content, "=")
    if ver_content is not None:
        res_dict["verification"] = parse_key_value_pairs(ver_content, "=")
    if crit_content is not None:
        res_dict["critique"] = parse_key_value_pairs(crit_content, ":")
    if res_content is not None:
        res_dict["research"] = parse_key_value_pairs(res_content, ":")
    if sr_content is not None:
        res_dict["self_review"] = parse_key_value_pairs(sr_content, ":")

    return res_dict


def expand_multimodal_placeholders(
    prompt: str,
    vision_budget: int = 280,
    audio_budget: int = 120,
) -> str:
    """
    Expands '<｜image｜>' and '<｜audio｜>' placeholders by repeating them
    according to the budget lengths.
    """
    prompt = prompt.replace("<｜image｜>", "<｜image｜>" * vision_budget)
    prompt = prompt.replace("<｜audio｜>", "<｜audio｜>" * audio_budget)
    return prompt
