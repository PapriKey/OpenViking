# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Ordinary-memory extraction must retain its caller's source archive scope."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import Message, TextPart
from openviking.server.identity import RequestContext, Role
from openviking.session.compressor_v3 import SessionCompressorV3
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "archive_uri",
    [
        None,
        "viking://user/u/sessions/s1/history/archive_001/",
        "viking://user/u/sessions/s1/history/archive_002",
    ],
)
async def test_user_memory_extraction_preserves_archive_scope(monkeypatch, tmp_path, archive_uri):
    previous_config = OpenVikingConfigSingleton._instance
    OpenVikingConfigSingleton.initialize(
        config_dict={
            "storage": {"workspace": str(tmp_path)},
            "embedding": {
                "dense": {"provider": "openai", "model": "test", "api_key": "test", "dimension": 8}
            },
        }
    )
    captured = {}
    registry = SimpleNamespace()

    def orchestrator(**kwargs):
        captured["context"] = kwargs["context_provider"].get_extract_context()
        return SimpleNamespace(run=AsyncMock(return_value=(None, [])))

    monkeypatch.setattr("openviking.session.compressor_v3.get_viking_fs", SimpleNamespace)
    monkeypatch.setattr("openviking.session.compressor_v3.get_default_registry", lambda: registry)
    monkeypatch.setattr(
        "openviking.session.memory.account_templates.resolve_account_memory_registry",
        AsyncMock(return_value=registry),
    )
    compressor = SessionCompressorV3(
        vikingdb=None,
        vlm_resolver=SimpleNamespace(get_vlm=AsyncMock(return_value=SimpleNamespace())),
    )
    compressor._get_or_create_react = orchestrator
    try:
        await compressor._extract_user_memories(
            messages=[Message(id="m1", role="user", parts=[TextPart("Remember this fact.")])],
            ctx=RequestContext(user=UserIdentifier.the_default_user("u"), role=Role.ROOT),
            archive_uri=archive_uri,
            allow_self_memory=False,
            allowed_memory_types={"events"},
            strict_extract_errors=True,
        )
        context = captured["context"]
        expected_uri = (archive_uri or "").rstrip("/")
        assert context.source_archive_uri == expected_uri
        expected_suffix = (
            "_" + hashlib.sha256(expected_uri.encode()).hexdigest()[:16] if expected_uri else ""
        )
        assert context.get_source_archive_suffix() == expected_suffix
    finally:
        if previous_config is None:
            OpenVikingConfigSingleton.reset_instance()
        else:
            OpenVikingConfigSingleton.set_instance(previous_config)
