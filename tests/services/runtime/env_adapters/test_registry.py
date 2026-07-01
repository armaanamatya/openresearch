from backend.services.runtime.asset_cache import AssetCache
from backend.services.runtime.env_adapters.registry import resolve_adapter
from backend.services.runtime.env_adapters.alfworld import AlfworldAdapter
from backend.services.runtime.env_adapters.webshop import WebShopAdapter
from backend.services.runtime.env_adapters.search_qa import SearchQaAdapter


def test_routes_by_name_and_alias(tmp_path):
    c = AssetCache(tmp_path)
    adapters = [AlfworldAdapter(c), WebShopAdapter(c), SearchQaAdapter(c)]
    assert isinstance(resolve_adapter("alf-world", adapters), AlfworldAdapter)
    assert isinstance(resolve_adapter("WebShop", adapters), WebShopAdapter)
    assert isinstance(resolve_adapter("searchqa", adapters), SearchQaAdapter)
    assert resolve_adapter("mnist", adapters) is None          # unknown → None
