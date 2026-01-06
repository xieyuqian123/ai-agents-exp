from typing import Any, Callable, Dict, Iterable, Optional


class ToolExecutor:
    def __init__(self, tools: Optional[Dict[str, Callable[..., Any]]] = None):
        self._tools: Dict[str, Callable[..., Any]] = dict(tools or {})

    def register(self, name: str, func: Callable[..., Any]) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("工具名称必须是非空字符串。")
        if not callable(func):
            raise ValueError("工具必须是可调用对象。")
        if name in self._tools:
            raise KeyError(f"工具已存在，禁止重复注册: {name}")
        self._tools[name] = func

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> Iterable[str]:
        return self._tools.keys()

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> Any:
        func = self._tools.get(name)
        if not func:
            raise KeyError(f"工具不存在: {name}")
        return func(**kwargs)
