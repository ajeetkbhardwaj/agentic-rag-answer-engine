import pytest
from agents.router_agent import RouterAgent


def test_internal_prefers_documents():
    r = RouterAgent()
    decision = r.decide("Show our internal logistics report", {"has_uploaded_docs": True})
    assert decision["use_documents"] is True
    assert decision["use_web"] is False


def test_web_needed_when_latest():
    r = RouterAgent()
    decision = r.decide("What are the latest supply chain trends in 2025?", {"has_uploaded_docs": False})
    assert decision["use_web"] is True


def test_default_uses_both_when_docs_present():
    r = RouterAgent()
    decision = r.decide("Compare vendor SLAs", {"has_uploaded_docs": True})
    assert decision["use_documents"] is True
