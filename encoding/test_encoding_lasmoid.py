"""
Test suite for Lasmoid Encoding (with V2 Cognitive Architecture Upgrades).
"""

import json
import os
import sys

# Add encoding directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from encoding_lasmoid import (
    encode_messages, 
    parse_message_from_completion_text, 
    validate_tokenizer_config,
    LasmoidTokenizer,
    LASMOID_TOKENIZER_PROFILE,
    REASONING_POLICY,
    MULTIMODAL_SP_TOKENS,
    bos_token,
    eos_token
)

TESTS_DIR = "/Users/abhishekjha/CODE/NEXUS/DeepSeek-V4-Pro/encoding/tests"

# ============================================================
# V1 Compatibility Tests
# ============================================================

def test_case_1():
    """Thinking mode with tool calls (multi-turn, tool results merged into user)."""
    with open(os.path.join(TESTS_DIR, "test_input_1.json")) as f:
        td = json.load(f)
        messages = td["messages"]
        messages[0]["tools"] = td["tools"]
    gold = open(os.path.join(TESTS_DIR, "test_output_1.txt")).read()
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # Parse: assistant turn with tool call
    marker = "<｜Assistant｜><think>"
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find("<｜User｜>", first_start)
    parsed_tc = parse_message_from_completion_text(prompt[first_start:first_end], thinking_mode="thinking")
    assert parsed_tc["reasoning_content"] == "The user wants to know the weather in Beijing. I should use the get_weather tool."
    assert parsed_tc["content"] == ""
    assert len(parsed_tc["tool_calls"]) == 1
    assert parsed_tc["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed_tc["tool_calls"][0]["function"]["arguments"]) == {"location": "Beijing", "unit": "celsius"}

    # Parse: final assistant turn with content
    last_start = prompt.rfind(marker) + len(marker)
    parsed_final = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    assert parsed_final["reasoning_content"] == "Got the weather data. Let me format a nice response."
    assert "22°C" in parsed_final["content"]
    assert parsed_final["tool_calls"] == []

    print("  [PASS] case 1: thinking with tools (encode + parse)")


def test_case_2():
    """Thinking mode without tools (drop_thinking removes earlier reasoning)."""
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_2.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_2.txt")).read()
    prompt = encode_messages(messages, thinking_mode="thinking")
    assert prompt == gold

    # Parse: last assistant turn
    marker = "<｜Assistant｜><think>"
    last_start = prompt.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[last_start:], thinking_mode="thinking")
    assert parsed["reasoning_content"] == "The user asks about the capital of France. It is Paris."
    assert parsed["content"] == "The capital of France is Paris."
    assert parsed["tool_calls"] == []

    # Verify drop_thinking: first assistant's reasoning should be absent
    assert "The user said hello" not in prompt

    print("  [PASS] case 2: thinking without tools (encode + parse)")


def test_case_3():
    """Interleaved thinking + search (developer with tools, latest_reminder)."""
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_3.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_3.txt")).read()
    assert encode_messages(messages, thinking_mode="thinking") == gold
    print("  [PASS] case 3: interleaved thinking + search")


def test_case_4():
    """Quick instruction task with latest_reminder (chat mode, action task)."""
    messages = json.load(open(os.path.join(TESTS_DIR, "test_input_4.json")))
    gold = open(os.path.join(TESTS_DIR, "test_output_4.txt")).read()
    assert encode_messages(messages, thinking_mode="chat") == gold
    print("  [PASS] case 4: quick instruction task")


# ============================================================
# V2 Cognitive OS Architecture Tests
# ============================================================

def test_case_v2_cognitive():
    """Test v2 cognitive reasoning mode (agentic) with structured internal blocks."""
    messages = [
        {"role": "user", "content": "How's the climate in Paris?"},
        {
            "role": "assistant",
            "working_memory": {
                "current_goal": "check Paris weather",
                "subtasks": "extract features"
            },
            "plan": {
                "goal": "determine climate",
                "constraints": "use metric units"
            },
            "evidence_check": {
                "known": "location=Paris",
                "unknown": "season"
            },
            "tool_decision": {
                "benefit": "high",
                "decision": "call"
            },
            "critique": {
                "assumptions": "standard temperature range"
            },
            "self_review": {
                "accuracy": "unverified"
            },
            "reasoning_content": "Paris is in France. We need local weather info.",
            "content": "Paris has a temperate climate."
        }
    ]

    prompt = encode_messages(messages, reasoning_mode="agentic", reasoning_effort="max")

    # 1. Verify reasoning policy prepended
    assert prompt.startswith(bos_token + REASONING_POLICY)

    # 2. Verify all tag blocks rendered correctly
    assert "<working_memory>\ncurrent_goal: check Paris weather\nsubtasks: extract features\n</working_memory>" in prompt
    assert "<plan>\ngoal: determine climate\nconstraints: use metric units\n</plan>" in prompt
    assert "<evidence_check>\nknown: location=Paris\nunknown: season\n</evidence_check>" in prompt
    assert "<tool_decision>\nbenefit=high\ndecision=call\n</tool_decision>" in prompt
    assert "<critique>\nassumptions: standard temperature range\n</critique>" in prompt
    assert "<self_review>\naccuracy: unverified\n</self_review>" in prompt
    assert "Paris is in France. We need local weather info." in prompt

    # 3. Verify parse extracts all tags successfully and cleans reasoning content
    marker = "<｜Assistant｜><think>"
    start_pos = prompt.find(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[start_pos:], thinking_mode="thinking")

    assert parsed["working_memory"] == {"current_goal": "check Paris weather", "subtasks": "extract features"}
    assert parsed["plan"] == {"goal": "determine climate", "constraints": "use metric units"}
    assert parsed["evidence_check"] == {"known": "location=Paris", "unknown": "season"}
    assert parsed["tool_decision"] == {"benefit": "high", "decision": "call"}
    assert parsed["critique"] == {"assumptions": "standard temperature range"}
    assert parsed["self_review"] == {"accuracy": "unverified"}
    
    # Check cleaned reasoning content (tags removed)
    assert parsed["reasoning_content"] == "Paris is in France. We need local weather info."
    assert parsed["content"] == "Paris has a temperate climate."

    print("  [PASS] case v2: cognitive OS architecture (encode + parse)")


def test_case_v2_parallel_tools():
    """Test v2 parallel tool calls with confidence attributes."""
    messages = [
        {"role": "user", "content": "Gather info from Google and Wikidata"},
        {
            "role": "assistant",
            "parallel": True,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "google_search",
                        "arguments": '{"query": "Paris weather"}'
                    },
                    "confidence": "0.95",
                    "reason": "need latest info",
                    "expected_information": "current temp"
                },
                {
                    "type": "function",
                    "function": {
                        "name": "wikidata_query",
                        "arguments": '{"entity": "Q90"}'
                    },
                    "confidence": "0.82",
                    "reason": "need coordinates",
                    "expected_information": "geo data"
                }
            ],
            "content": ""
        }
    ]

    prompt = encode_messages(messages, reasoning_mode="fast")

    # Verify parallel block name
    assert "<｜DSML｜parallel_tool_calls>" in prompt
    assert "</｜DSML｜parallel_tool_calls>" in prompt

    # Verify attributes in invoke tag
    assert '<｜DSML｜invoke name="google_search" confidence="0.95" reason="need latest info" expected_information="current temp">' in prompt
    assert '<｜DSML｜invoke name="wikidata_query" confidence="0.82" reason="need coordinates" expected_information="geo data">' in prompt

    # Parse back and verify attributes
    marker = "<｜Assistant｜></think>"
    start_pos = prompt.find(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt[start_pos:], thinking_mode="chat")

    assert len(parsed["tool_calls"]) == 2
    assert parsed["tool_calls"][0]["confidence"] == "0.95"
    assert parsed["tool_calls"][0]["reason"] == "need latest info"
    assert parsed["tool_calls"][0]["expected_information"] == "current temp"
    assert parsed["tool_calls"][0]["function"]["name"] == "google_search"

    assert parsed["tool_calls"][1]["confidence"] == "0.82"
    assert parsed["tool_calls"][1]["reason"] == "need coordinates"
    assert parsed["tool_calls"][1]["expected_information"] == "geo data"
    assert parsed["tool_calls"][1]["function"]["name"] == "wikidata_query"

    print("  [PASS] case v2: parallel tool calls (encode + parse)")


def test_case_v2_structured_results():
    """Test v2 structured JSON tool results inside user content blocks."""
    messages = [
        {"role": "user", "content": "Fetch Paris weather"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool": "web_search",
            "status": "success",
            "confidence": 0.94,
            "results": [
                {
                    "source": "wikipedia",
                    "relevance": 0.98,
                    "freshness": 0.9,
                    "trustworthiness": 0.95,
                    "content": "Paris temperature is 22C."
                }
            ]
        }
    ]

    prompt = encode_messages(messages, reasoning_mode="fast")

    # Verify structured tool result JSON exists inside <tool_result> tag
    assert "<tool_result>" in prompt
    assert "</tool_result>" in prompt
    
    start_tag = "<tool_result>"
    end_tag = "</tool_result>"
    s_idx = prompt.find(start_tag) + len(start_tag)
    e_idx = prompt.find(end_tag, s_idx)
    json_str = prompt[s_idx:e_idx].strip()
    
    data = json.loads(json_str)
    assert data["tool"] == "web_search"
    assert data["status"] == "success"
    assert data["confidence"] == 0.94
    assert data["results"][0]["source"] == "wikipedia"
    assert data["results"][0]["relevance"] == 0.98

    print("  [PASS] case v2: structured JSON tool results (encode + parse)")


def test_case_v3_tokenizer_contract():
    """Tokenizer config exposes the long-context Lasmoid special-token contract."""
    with open("/Users/abhishekjha/CODE/NEXUS/Lasmoid/configs/tokenizer/tokenizer_config.json") as f:
        cfg = json.load(f)
    with open("/Users/abhishekjha/CODE/NEXUS/Lasmoid/tokenizer.json") as f:
        tokenizer_json = json.load(f)

    report = validate_tokenizer_config(cfg)
    assert report["ok"], report
    assert cfg["model_max_length"] >= 1048576
    assert "<｜User｜>" in report["required_special_tokens"]
    assert MULTIMODAL_SP_TOKENS["embedding"] in report["required_special_tokens"]
    added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
    assert cfg["vocab_size"] == max(added.values()) + 1
    for token in report["required_special_tokens"]:
        assert token in added, token
    assert "<answer>" in added

    print("  [PASS] case v3: tokenizer contract validation")


def test_case_v3_multimodal_and_embeddings():
    """List content blocks render into deterministic frontier placeholders."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this evidence."},
                {"type": "image_url", "image_url": {"url": "https://example.test/scan.png"}},
                {"type": "file", "name": "brief.pdf", "mime_type": "application/pdf", "file_id": "file_123"},
                {"type": "semantic_embedding", "model": "lasmoid-retrieval", "vector_id": "vec_9", "dim": 4096},
            ],
        }
    ]

    prompt = encode_messages(messages, reasoning_mode="fast")
    assert "Inspect this evidence." in prompt
    assert MULTIMODAL_SP_TOKENS["image"] in prompt
    assert MULTIMODAL_SP_TOKENS["file"] in prompt
    assert MULTIMODAL_SP_TOKENS["embedding"] in prompt
    assert '"vector_id": "vec_9"' in prompt

    print("  [PASS] case v3: multimodal and embedding placeholders")


def test_case_v3_escaped_tool_arguments_round_trip():
    """DSML escaping preserves XML-like argument text and attributes."""
    messages = [
        {"role": "user", "content": "Call the formatter."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "format_payload",
                        "arguments": json.dumps({
                            "snippet": "<tag a=\"1\">A & B</tag>",
                            "count": 2,
                        }),
                    },
                    "reason": "needs <xml> escaping",
                }
            ],
            "content": "",
        },
    ]

    prompt = encode_messages(messages, reasoning_mode="fast")
    assert "&lt;tag a=\"1\"&gt;A &amp; B&lt;/tag&gt;" in prompt
    marker = "<｜Assistant｜></think>"
    parsed = parse_message_from_completion_text(prompt[prompt.find(marker) + len(marker):], thinking_mode="chat")
    call = parsed["tool_calls"][0]
    assert call["reason"] == "needs <xml> escaping"
    assert json.loads(call["function"]["arguments"]) == {
        "snippet": "<tag a=\"1\">A & B</tag>",
        "count": 2,
    }

    print("  [PASS] case v3: escaped tool arguments round trip")


# ============================================================
# Tokenizer Round-Trip & ID-Range Tests (Req 17.1, 17.2)
# ============================================================

LASMOID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_tokenizer_roundtrip():
    """Encode→decode reproduces in-vocabulary text (Req 17.1)."""
    tok = LasmoidTokenizer(LASMOID_DIR)

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def foo(x): return x + 1",
        "Unicode: こんにちは世界 🌍",
        "1234567890",
        "A" * 100,
        " ",
        "a",
        "mixed CaSe TeXt",
        "Punctuation: !@#$%^&*()_+-=[]{}|;:,.<>?",
        "Newlines\nand\ttabs",
        "Repeated   spaces   here",
        "Code: if __name__ == '__main__': print('hi')",
    ]

    for text in test_texts:
        result = tok.roundtrip(text)
        assert result == text, (
            f"Roundtrip failed for {repr(text[:50])}: got {repr(result[:50])}"
        )

    print("  [PASS] tokenizer roundtrip (Req 17.1)")


def test_tokenizer_batch_encode_id_range():
    """Batch encoding yields ids within [0, vocab_size) (Req 17.2)."""
    tok = LasmoidTokenizer(LASMOID_DIR)
    vocab_size = tok.vocab_size
    assert vocab_size == LASMOID_TOKENIZER_PROFILE.vocab_size

    test_texts = [
        "Hello world",
        "Testing batch encode decode",
        "Lasmoid tokenizer validation",
        "Mixed 123 numbers and symbols @#$",
        "A longer passage of text that exercises more of the vocabulary space "
        "with various words and character patterns.",
    ]

    batch = tok.batch_encode(test_texts)
    assert len(batch) == len(test_texts)

    for i, ids in enumerate(batch):
        assert len(ids) > 0, f"Empty encoding for text at index {i}"
        for token_id in ids:
            assert 0 <= token_id < vocab_size, (
                f"Token id {token_id} out of range [0, {vocab_size}) "
                f"in batch element {i}"
            )

    print("  [PASS] batch encode id range [0, vocab_size) (Req 17.2)")


def test_tokenizer_vocab_size_matches_config():
    """TokenizerProfile.vocab_size matches the actual tokenizer config."""
    with open(os.path.join(LASMOID_DIR, "configs", "tokenizer", "tokenizer_config.json")) as f:
        cfg = json.load(f)

    assert LASMOID_TOKENIZER_PROFILE.vocab_size == cfg["vocab_size"], (
        f"TokenizerProfile.vocab_size={LASMOID_TOKENIZER_PROFILE.vocab_size} "
        f"!= tokenizer_config.json vocab_size={cfg['vocab_size']}"
    )

    tok = LasmoidTokenizer(LASMOID_DIR)
    assert tok.vocab_size == cfg["vocab_size"]

    print("  [PASS] vocab_size consistency check")


if __name__ == "__main__":
    print("Running Lasmoid Encoding & Cognitive OS Tests...\n")
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_v2_cognitive()
    test_case_v2_parallel_tools()
    test_case_v2_structured_results()
    test_case_v3_tokenizer_contract()
    test_case_v3_multimodal_and_embeddings()
    test_case_v3_escaped_tool_arguments_round_trip()
    test_tokenizer_roundtrip()
    test_tokenizer_batch_encode_id_range()
    test_tokenizer_vocab_size_matches_config()
    print("\nAll tests passed successfully!")
