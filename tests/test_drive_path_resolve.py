"""resolve_personal_path — the drive-path turn fact's resolver (2026-09-27).

One READY personal asset at the exact named path, or None — never a guess.
Also pins the chat-context token extractor that feeds it.
"""
from uuid import uuid4

from core.application.chat.context import _PATH_TOKEN, _attach_asset_id
from core.application.drive_service import READY
from tests._drive_fakes import make_drive

import types


async def test_resolve_exact_personal_path(tmp_path):
    svc = make_drive(tmp_path)
    owner, other = uuid4(), uuid4()
    a = await svc.assets.create(owner, "如何炒西红柿_v1.pdf",
                                folder_path="Reseach/如何炒西红柿/outputs",
                                file_status=READY)
    # prefix-tolerant: "My Drive/…" and the bare stored form resolve the same
    found = await svc.resolve_personal_path(
        owner, "My Drive/Reseach/如何炒西红柿/outputs/如何炒西红柿_v1.pdf")
    assert found == {"asset_id": str(a.id), "name": "如何炒西红柿_v1.pdf",
                     "folder_path": "Reseach/如何炒西红柿/outputs"}
    assert (await svc.resolve_personal_path(
        owner, "Reseach/如何炒西红柿/outputs/如何炒西红柿_v1.pdf"))["asset_id"] == str(a.id)
    # another user's same-named path is NOT this user's fact (personal scope)
    assert await svc.resolve_personal_path(
        other, "My Drive/Reseach/如何炒西红柿/outputs/如何炒西红柿_v1.pdf") is None


async def test_resolve_refuses_ambiguity_and_unready(tmp_path):
    svc = make_drive(tmp_path)
    u = uuid4()
    await svc.assets.create(u, "notes.md", folder_path="a", file_status=READY)
    # not-found path → None (never fuzzy)
    assert await svc.resolve_personal_path(u, "My Drive/a/other.md") is None
    assert await svc.resolve_personal_path(u, "My Drive/b/notes.md") is None
    # root file resolves with folder_path None
    r = await svc.assets.create(u, "readme.md", folder_path=None, file_status=READY)
    assert (await svc.resolve_personal_path(u, "My Drive/readme.md"))["asset_id"] == str(r.id)
    # uploading/not-ready asset is not a settled fact
    p = await svc.assets.create(u, "big.pdf", folder_path="a", file_status="uploading")
    assert await svc.resolve_personal_path(u, "My Drive/a/big.pdf") is None
    # junk paths
    assert await svc.resolve_personal_path(u, "My Drive/") is None
    assert await svc.resolve_personal_path(u, "") is None


def test_path_token_extractor_is_conservative():
    m = _PATH_TOKEN.search("帮我阅读 My Drive/Reseach/如何炒西红柿/outputs/如何炒西红柿_v1.pdf 这份文档")
    assert m and m.group(1) == "Reseach/如何炒西红柿/outputs/如何炒西红柿_v1.pdf"
    # bare filename is NOT a path fact
    assert _PATH_TOKEN.search("读取 report.pdf") is None
    assert _PATH_TOKEN.search("总结这份文档") is None
    # CJK root form + trailing sentence punctuation excluded from the token
    m = _PATH_TOKEN.search("打开我的云盘/a/b.txt，谢谢")
    assert m and m.group(1) == "a/b.txt"


def test_attach_asset_id_helper():
    assert _attach_asset_id(types.SimpleNamespace(attach={"asset_id": "x"})) == "x"
    assert _attach_asset_id(types.SimpleNamespace(
        attach=types.SimpleNamespace(asset_id="y"))) == "y"
    assert _attach_asset_id(types.SimpleNamespace(attach=None)) == ""
    assert _attach_asset_id(types.SimpleNamespace(attach={"owned": True})) == ""
