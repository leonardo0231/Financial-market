import httpx
from typing import Any, Dict, Optional
from ..settings import settings

DEFAULT_TIMEOUT = 30.0

async def get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()

async def post_json(url: str, json_body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post(url, json=json_body, headers=headers)
        r.raise_for_status()
        if r.content and r.headers.get("content-type","").startswith("application/json"):
            return r.json()
        return {}