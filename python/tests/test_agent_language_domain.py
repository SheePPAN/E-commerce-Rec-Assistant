import re
from pathlib import Path


CHINESE_OR_YUAN = re.compile(r"[\u4e00-\u9fff]|¥")
PYTHON_ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = [
    PYTHON_ROOT / "agents" / "product_rec_agent.py",
    PYTHON_ROOT / "agents" / "user_profile_agent.py",
    PYTHON_ROOT / "agents" / "marketing_copy_agent.py",
]


def assert_english_domain_text(value: str) -> None:
    assert not CHINESE_OR_YUAN.search(value)


def agent_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in AGENT_FILES)


def test_agent_prompts_are_english_amazon_electronics_prompts():
    prompt_text = agent_source()

    assert "Amazon Electronics" in prompt_text
    assert "review highlights" in prompt_text
    assert_english_domain_text(prompt_text)


def test_fallback_products_match_amazon_electronics_domain():
    source = (PYTHON_ROOT / "agents" / "product_rec_agent.py").read_text(encoding="utf-8")

    assert "MOCK_PRODUCTS" in source
    assert "AMZ-FALLBACK-" in source
    assert "amazon-electronics-fallback" in source
    assert "Noise Cancelling Bluetooth Headphones" in source
    assert "Cell Phones & Accessories" in source
    assert_english_domain_text(source)


def test_marketing_copy_no_longer_uses_chinese_ad_law_filter():
    source = (PYTHON_ROOT / "agents" / "marketing_copy_agent.py").read_text(encoding="utf-8")

    assert "FORBIDDEN_WORDS" not in source
    assert "_compliance_check" not in source
    assert "Chinese Ad Law" not in source


def test_recall_strategy_is_reported_from_actual_recall_path():
    source = (PYTHON_ROOT / "agents" / "product_rec_agent.py").read_text(encoding="utf-8")
    supervisor_source = (PYTHON_ROOT / "orchestrator" / "supervisor.py").read_text(
        encoding="utf-8"
    )
    graph_source = (PYTHON_ROOT / "orchestrator" / "graph.py").read_text(
        encoding="utf-8"
    )

    assert "SEMANTIC_RECALL_STRATEGY" in source
    assert "FALLBACK_RECALL_STRATEGY" in source
    assert "candidates, recall_strategy = await self._recall" in source
    assert "return products, SEMANTIC_RECALL_STRATEGY" in source
    assert "return candidates[:limit], FALLBACK_RECALL_STRATEGY" in source
    assert "recall_strategy=recall_strategy" in source
    assert 'recall_strategy=getattr(rec_result, "recall_strategy", "")' in supervisor_source
    assert 'state["recall_strategy"] = getattr(result, "recall_strategy", "")' in graph_source
