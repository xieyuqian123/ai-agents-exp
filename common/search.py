import os
from typing import Any, Dict, List, Optional

from serpapi import GoogleSearch


def serpapi_search_raw(
    query: str,
    *,
    api_key: Optional[str] = None,
    engine: str = "google",
    num_results: int = 10,
    location: Optional[str] = None,
    hl: Optional[str] = None,
    gl: Optional[str] = None,
    safe: Optional[str] = None,
    start: Optional[int] = None,
    **extra_params: Any,
) -> Dict[str, Any]:
    key = api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not key:
        raise ValueError("SerpApi API key 未配置，请设置 SERPAPI_API_KEY 环境变量或传入 api_key。")

    params: Dict[str, Any] = {"engine": engine, "q": query, "api_key": key}

    if num_results is not None:
        params["num"] = int(num_results)
    if location:
        params["location"] = location
    if hl:
        params["hl"] = hl
    if gl:
        params["gl"] = gl
    if safe:
        params["safe"] = safe
    if start is not None:
        params["start"] = int(start)

    params.update(extra_params)

    search = GoogleSearch(params)
    return search.get_dict()


def extract_organic_results(payload: Dict[str, Any], *, limit: int = 5) -> List[Dict[str, Any]]:
    organic = payload.get("organic_results") or []
    results: List[Dict[str, Any]] = []
    for item in organic[: max(0, int(limit))]:
        results.append(
            {
                "position": item.get("position"),
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "source": item.get("source"),
                "displayed_link": item.get("displayed_link"),
            }
        )
    return results


def serpapi_search(
    query: str,
    *,
    api_key: Optional[str] = None,
    limit: int = 5,
    engine: str = "google",
    location: Optional[str] = None,
    hl: Optional[str] = None,
    gl: Optional[str] = None,
    safe: Optional[str] = None,
    start: Optional[int] = None,
    **extra_params: Any,
) -> List[Dict[str, Any]]:
    payload = serpapi_search_raw(
        query,
        api_key=api_key,
        engine=engine,
        num_results=max(int(limit), 1),
        location=location,
        hl=hl,
        gl=gl,
        safe=safe,
        start=start,
        **extra_params,
    )
    return extract_organic_results(payload, limit=limit)


def serpapi_search_text(
    query: str,
    *,
    api_key: Optional[str] = None,
    limit: int = 5,
    engine: str = "google",
    location: Optional[str] = None,
    hl: Optional[str] = None,
    gl: Optional[str] = None,
    safe: Optional[str] = None,
    start: Optional[int] = None,
    **extra_params: Any,
) -> str:
    payload = serpapi_search_raw(
        query,
        api_key=api_key,
        engine=engine,
        num_results=max(int(limit), 1),
        location=location,
        hl=hl,
        gl=gl,
        safe=safe,
        start=start,
        **extra_params,
    )

    answer_box_list = payload.get("answer_box_list")
    if isinstance(answer_box_list, list) and answer_box_list:
        items = [str(x).strip() for x in answer_box_list if str(x).strip()]
        if items:
            return "\n".join(items)

    answer_box = payload.get("answer_box")
    if isinstance(answer_box, dict):
        for k in ("answer", "snippet", "result", "title"):
            v = answer_box.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    knowledge_graph = payload.get("knowledge_graph")
    if isinstance(knowledge_graph, dict):
        for k in ("description", "title", "type"):
            v = knowledge_graph.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    results = extract_organic_results(payload, limit=limit)
    if not results:
        return f"对不起，没有找到关于 '{query}' 的信息。"

    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        link = (r.get("link") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if snippet:
            lines.append(f"{i}. {title}\n{link}\n{snippet}")
        else:
            lines.append(f"{i}. {title}\n{link}")
    return "\n\n".join(lines)
