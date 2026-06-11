"""MarkdownStore のテスト。"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from engram.store import MarkdownStore, content_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return MarkdownStore(tmp_path / "memories")


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

def test_content_hash_consistent():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("  hello  ") == content_hash("hello")


def test_content_hash_different():
    assert content_hash("hello") != content_hash("world")


# ---------------------------------------------------------------------------
# create: サブディレクトリ振り分け
# ---------------------------------------------------------------------------

def test_create_knowledge_subdir(store):
    r = store.create(
        content="Knowledge about Python",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    assert r.path is not None
    assert "knowledge" in r.path.parts
    assert r.path.exists()


def test_create_preference_subdir(store):
    r = store.create(
        content="I prefer dark mode",
        type="preference",
        importance=3,
        created="2026-06-11T09:00:00+09:00",
    )
    assert "preferences" in r.path.parts


def test_create_project_subdir(store):
    r = store.create(
        content="Project KnowledgeBase plan",
        type="project",
        importance=7,
        created="2026-06-11T09:00:00+09:00",
    )
    assert "projects" in r.path.parts


def test_create_episode_subdir_by_year_month(store):
    r = store.create(
        content="Episode from June 2026",
        type="episode",
        importance=4,
        created="2026-06-11T09:00:00+09:00",
    )
    # episodes/2026/06
    parts = r.path.parts
    assert "episodes" in parts
    ep_idx = parts.index("episodes")
    assert parts[ep_idx + 1] == "2026"
    assert parts[ep_idx + 2] == "06"


def test_create_episode_different_month(store):
    r = store.create(
        content="Episode from January 2025",
        type="episode",
        importance=2,
        created="2025-01-15T12:00:00+09:00",
    )
    parts = r.path.parts
    ep_idx = parts.index("episodes")
    assert parts[ep_idx + 1] == "2025"
    assert parts[ep_idx + 2] == "01"


# ---------------------------------------------------------------------------
# frontmatter 往復(links の [[id]] 変換含む)
# ---------------------------------------------------------------------------

def test_frontmatter_roundtrip_basic(store):
    r = store.create(
        content="Round trip test content",
        type="knowledge",
        importance=6,
        tags=["test", "roundtrip"],
        source="claude-code",
        created="2026-06-11T09:00:00+09:00",
    )
    restored = store.read(r.path)
    assert restored.id == r.id
    assert restored.type == "knowledge"
    assert restored.importance == 6
    assert restored.tags == ["test", "roundtrip"]
    assert restored.source == "claude-code"
    assert restored.tier == "hot"
    assert restored.content.strip() == "Round trip test content"


def test_links_wiki_link_roundtrip(store):
    r = store.create(
        content="Memory with links",
        type="knowledge",
        importance=5,
        links=["01JAAABBBCCC000111", "01JDDDEEEFFF222333"],
        created="2026-06-11T09:00:00+09:00",
    )
    # On disk should be [[...]] format
    text = r.path.read_text(encoding="utf-8")
    assert "[[01JAAABBBCCC000111]]" in text

    # When read back, links should be plain ids
    restored = store.read(r.path)
    assert "01JAAABBBCCC000111" in restored.links
    assert "01JDDDEEEFFF222333" in restored.links
    # No brackets in the record
    for lid in restored.links:
        assert "[[" not in lid
        assert "]]" not in lid


def test_links_empty_roundtrip(store):
    r = store.create(
        content="No links memory",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    restored = store.read(r.path)
    assert restored.links == []


# ---------------------------------------------------------------------------
# ファイル名規約と日本語 slug
# ---------------------------------------------------------------------------

def test_filename_format(store):
    r = store.create(
        content="Hello World content",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    name = r.path.name
    # YYYYMMDD-{slug}-{6chars}.md
    assert name.endswith(".md")
    parts = name[:-3].split("-")
    assert parts[0] == "20260611"
    # Last part should be 6 chars (id suffix)
    assert len(parts[-1]) == 6


def test_filename_japanese_slug(store):
    r = store.create(
        content="日本語のメモです",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    name = r.path.name
    assert "日本語のメモです" in name or "日本語" in name


def test_filename_windows_forbidden_chars_removed(store):
    r = store.create(
        content='Test <content> with "forbidden" chars: ok',
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    name = r.path.name
    for ch in '<>:"/\\|?*':
        assert ch not in name


def test_filename_slug_max_40_chars(store):
    long_content = "A" * 100 + " more text"
    r = store.create(
        content=long_content,
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    name = r.path.name
    # name = YYYYMMDD-{slug}-{6}.md; slug portion between first and last dash
    # The slug is at most 40 chars
    # Strip .md and split on -
    stem = name[:-3]
    # date part is 8 chars, last part is 6 chars
    middle = stem[9:-7]  # remove "YYYYMMDD-" and "-XXXXXX"
    assert len(middle) <= 40


def test_filename_spaces_to_dashes(store):
    r = store.create(
        content="hello world space test",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    name = r.path.name
    assert " " not in name


# ---------------------------------------------------------------------------
# update / content_hash 変化
# ---------------------------------------------------------------------------

def test_update_changes_content_hash(store):
    r = store.create(
        content="Original content here",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    original_hash = r.content_hash

    from dataclasses import replace
    updated_record = replace(r, content="Updated content here - completely different")
    updated = store.update(updated_record)

    assert updated.content_hash != original_hash
    assert updated.content == "Updated content here - completely different"
    # File should reflect the new content
    restored = store.read(updated.path)
    assert restored.content.strip() == "Updated content here - completely different"


def test_update_same_content_same_hash(store):
    r = store.create(
        content="Same content",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    from dataclasses import replace
    same = replace(r, content="Same content")
    updated = store.update(same)
    assert updated.content_hash == r.content_hash


# ---------------------------------------------------------------------------
# add_link: 重複無視
# ---------------------------------------------------------------------------

def test_add_link_no_duplicate(store):
    r = store.create(
        content="Link test memory",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    r2 = store.add_link(r, "TARGET_ID_001")
    r3 = store.add_link(r2, "TARGET_ID_001")  # duplicate

    assert r3.links.count("TARGET_ID_001") == 1


def test_add_link_multiple_different(store):
    r = store.create(
        content="Link test memory",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    r2 = store.add_link(r, "ID_001")
    r3 = store.add_link(r2, "ID_002")
    assert len(r3.links) == 2
    assert "ID_001" in r3.links
    assert "ID_002" in r3.links


# ---------------------------------------------------------------------------
# set_tier
# ---------------------------------------------------------------------------

def test_set_tier_updates_file(store):
    r = store.create(
        content="Tier test content",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    assert r.tier == "hot"
    updated = store.set_tier(r, "cold")
    assert updated.tier == "cold"
    restored = store.read(updated.path)
    assert restored.tier == "cold"


# ---------------------------------------------------------------------------
# move_to_trash
# ---------------------------------------------------------------------------

def test_move_to_trash(store):
    r = store.create(
        content="To be trashed content",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    original_path = r.path
    trashed = store.move_to_trash(r)

    assert not original_path.exists()
    assert trashed.path.exists()
    assert "_trash" in trashed.path.parts
    assert trashed.tier == "trash"
    restored = store.read(trashed.path)
    assert restored.tier == "trash"


# ---------------------------------------------------------------------------
# scan_all: _trash を除外
# ---------------------------------------------------------------------------

def test_scan_all_excludes_trash(store):
    r1 = store.create(
        content="Normal memory one",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    r2 = store.create(
        content="Normal memory two",
        type="episode",
        importance=3,
        created="2026-06-11T09:00:00+09:00",
    )
    r3 = store.create(
        content="Will be trashed soon",
        type="knowledge",
        importance=2,
        created="2026-06-11T09:00:00+09:00",
    )
    store.move_to_trash(r3)

    records = list(store.scan_all())
    ids = [r.id for r in records]
    assert r1.id in ids
    assert r2.id in ids
    assert r3.id not in ids


def test_scan_all_empty_store(store):
    records = list(store.scan_all())
    assert records == []


# ---------------------------------------------------------------------------
# 壊れた frontmatter をスキップして落ちない
# ---------------------------------------------------------------------------

def test_scan_all_skips_broken_frontmatter(store):
    r = store.create(
        content="Good memory here",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    # Write a broken file into the store
    broken = store._root / "knowledge" / "broken.md"
    broken.write_text("---\nthis: is: broken: yaml: {{\n---\ncontent", encoding="utf-8")

    # Should not raise, should yield the good record, skip broken
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        records = list(store.scan_all())

    ids = [rec.id for rec in records]
    assert r.id in ids
    # broken file was skipped (no record for it since it has no valid id)
    # warnings may or may not be emitted depending on yaml behavior


def test_scan_all_skips_no_frontmatter(store):
    """Plain markdown with no frontmatter should be skipped gracefully."""
    plain = store._root / "knowledge" / "plain.md"
    plain.write_text("Just some plain text, no frontmatter at all.", encoding="utf-8")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        records = list(store.scan_all())
    # Should not crash


# ---------------------------------------------------------------------------
# find_by_id
# ---------------------------------------------------------------------------

def test_find_by_id_found(store):
    r = store.create(
        content="Findable content",
        type="knowledge",
        importance=5,
        created="2026-06-11T09:00:00+09:00",
    )
    found = store.find_by_id(r.id)
    assert found is not None
    assert found.id == r.id


def test_find_by_id_not_found(store):
    result = store.find_by_id("NONEXISTENT_ID_HERE")
    assert result is None


# ---------------------------------------------------------------------------
# content_hash 関数
# ---------------------------------------------------------------------------

def test_content_hash_whitespace_normalized():
    h1 = content_hash("  hello world  ")
    h2 = content_hash("hello world")
    assert h1 == h2


def test_content_hash_is_hex(store):
    h = content_hash("some text")
    assert len(h) == 64
    int(h, 16)  # should not raise
