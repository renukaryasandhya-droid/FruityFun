from fruity_fun.agent import FruityFunAgent
from fruity_fun.config import Settings
from fruity_fun.retrieval import HybridRetriever


def test_agent_degrades_without_credentials():
    settings = Settings(_env_file=None, openai_api_key="", pinecone_api_key="")
    agent = FruityFunAgent(settings, HybridRetriever(settings, []))
    result = agent.invoke("Draw three happy fruits")

    assert "OpenAI key" in result["answer"]
    assert result["confidence"] == 0
    assert result["grounded"] is False
    assert result["image_path"] is None


def test_unsafe_prompt_is_redirected():
    settings = Settings(_env_file=None, openai_api_key="", pinecone_api_key="")
    agent = FruityFunAgent(settings, HybridRetriever(settings, []))
    result = agent.invoke("Draw fruit with gore")

    assert result["is_safe"] is False
    assert "cheerful and safe" in result["answer"]
    assert "fruit picnic" in result["image_prompt"]
