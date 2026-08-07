from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path


BUSINESS_MODULES = {
    "documents",
    "evaluation",
    "findings",
    "jobs",
    "optimization",
    "retrieval",
    "review",
    "rule_management",
}
DOMAIN_FORBIDDEN_PREFIXES = (
    "alembic",
    "fastapi",
    "langgraph",
    "minio",
    "pymysql",
    "sqlalchemy",
    "uvicorn",
    "tender_review.api",
    "tender_review.bootstrap",
    "tender_review.infrastructure",
    "tender_review.worker",
)
PORT_FORBIDDEN_ANNOTATIONS = {"Any", "Engine", "Minio", "Session"}


def load_python_sources(package_root: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        sources[".".join(parts)] = path.read_text(encoding="utf-8")
    return sources


def domain_dependency_violations(sources: Mapping[str, str]) -> list[str]:
    violations: list[str] = []
    modules = set(sources)
    for module, source in sources.items():
        if not _is_domain_module(module):
            continue
        for target in _imports(module, source, modules):
            if target.startswith(DOMAIN_FORBIDDEN_PREFIXES):
                violations.append(f"{module} -> {target}")
    return sorted(violations)


def cross_module_internal_violations(sources: Mapping[str, str]) -> list[str]:
    violations: list[str] = []
    modules = set(sources)
    for module, source in sources.items():
        source_area = _business_area(module)
        if source_area is None:
            continue
        for target in _imports(module, source, modules):
            target_area = _business_area(target)
            if target_area is None or target_area == source_area:
                continue
            target_parts = target.split(".")
            is_public = len(target_parts) == 2 or target_parts[2] == "public"
            if not is_public:
                violations.append(f"{module} -> {target}")
    return sorted(violations)


def dependency_cycles(sources: Mapping[str, str]) -> list[tuple[str, ...]]:
    modules = set(sources)
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, source in sources.items():
        for target in _imports(module, source, modules):
            resolved = _nearest_module(target, modules)
            if resolved is not None and resolved != module:
                graph[module].add(resolved)

    cycles: set[tuple[str, ...]] = set()
    active: list[str] = []
    active_set: set[str] = set()
    complete: set[str] = set()

    def visit(module: str) -> None:
        if module in complete:
            return
        if module in active_set:
            index = active.index(module)
            cycles.add(_canonical_cycle(tuple(active[index:])))
            return
        active.append(module)
        active_set.add(module)
        for target in sorted(graph[module]):
            visit(target)
        active.pop()
        active_set.remove(module)
        complete.add(module)

    for module in sorted(modules):
        visit(module)
    return sorted(cycles)


def port_annotation_violations(sources: Mapping[str, str]) -> list[str]:
    violations: list[str] = []
    for module, source in sources.items():
        if not (module.endswith(".ports") or module == "tender_review.worker.contracts"):
            continue
        tree = ast.parse(source, filename=module)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            annotations = [
                argument.annotation
                for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                if argument.annotation is not None
            ]
            if node.returns is not None:
                annotations.append(node.returns)
            for annotation in annotations:
                rendered = ast.unparse(annotation)
                names = {
                    child.id for child in ast.walk(annotation) if isinstance(child, ast.Name)
                }
                if "dict[" in rendered or names & PORT_FORBIDDEN_ANNOTATIONS:
                    violations.append(f"{module}.{node.name}: {rendered}")
    return sorted(violations)


def _imports(module: str, source: str, modules: set[str]) -> set[str]:
    targets: set[str] = set()
    tree = ast.parse(source, filename=module)
    is_package = any(candidate.startswith(module + ".") for candidate in modules)
    package = module.split(".") if is_package else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - node.level + 1
                base = package[: max(keep, 0)]
                target = ".".join(base + ((node.module or "").split(".")))
            else:
                target = node.module or ""
            if target:
                targets.add(target.rstrip("."))
    return targets


def _business_area(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "tender_review" and parts[1] in BUSINESS_MODULES:
        return parts[1]
    return None


def _is_domain_module(module: str) -> bool:
    parts = module.split(".")
    return _business_area(module) is not None and (
        parts[-1] == "models" or "domain" in parts[2:]
    )


def _nearest_module(target: str, modules: set[str]) -> str | None:
    parts = target.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))
