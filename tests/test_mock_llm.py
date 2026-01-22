from llm.factory import LLMFactory, set_llm, get_llm
import os


def test_create_mock_llm():
    # Temporarily set env to mock provider
    os.environ['LLM_PROVIDER'] = 'mock'
    llm = LLMFactory.create_llm(provider='mock')
    assert llm is not None
    resp = llm.chat([{'role':'user','content':'hello mock'}])
    assert 'MOCK_CHAT_REPLY' in resp

