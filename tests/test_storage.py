"""Tests for storage backends."""

import pytest
from emms.storage.base import InMemoryStore, JSONFileStore


class TestInMemoryStore:
    def test_save_load(self):
        store = InMemoryStore()
        store.save("key1", {"data": 42})
        assert store.load("key1") == {"data": 42}

    def test_load_missing(self):
        store = InMemoryStore()
        assert store.load("nope") is None

    def test_keys(self):
        store = InMemoryStore()
        store.save("a", 1)
        store.save("b", 2)
        assert sorted(store.keys()) == ["a", "b"]


class TestJSONFileStore:
    def test_roundtrip(self, tmp_path):
        store = JSONFileStore(tmp_path / "store")
        store.save("test", {"hello": "world", "num": 42})
        loaded = store.load("test")
        assert loaded["hello"] == "world"
        assert loaded["num"] == 42

    def test_load_missing(self, tmp_path):
        store = JSONFileStore(tmp_path / "store")
        assert store.load("nope") is None

    def test_keys(self, tmp_path):
        store = JSONFileStore(tmp_path / "store")
        store.save("alpha", {})
        store.save("beta", {})
        assert sorted(store.keys()) == ["alpha", "beta"]

    def test_creates_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        store = JSONFileStore(deep)
        store.save("test", [1, 2, 3])
        assert deep.exists()
