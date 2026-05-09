# Issue #17 LocalDocker Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP sandbox substrate by introducing a stable `RuntimeBackend` abstraction and a `LocalDockerBackend` implementation with predictable sandbox lifecycle and shared-volume conventions.

**Architecture:** Add a small Python package under `src/reprolab/sandbox/` that defines backend models, the abstract runtime interface, and a Docker-backed implementation using an injectable client adapter. Keep the backend self-contained because dependency `#7` is not present in the current tree, and encode the PRD mount strategy directly in config and tests so later runtime tickets can integrate against a stable abstraction.

**Tech Stack:** Python 3.12, pytest, dataclasses, pathlib, async interfaces, Docker SDK integration via optional import/injection.

---

### Task 1: Scaffold the sandbox package and its test harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/reprolab/__init__.py`
- Create: `src/reprolab/sandbox/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing test harness check**

```python
def test_package_imports() -> None:
    import reprolab.sandbox  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sandbox/test_local_docker_backend.py::test_package_imports -v`
Expected: FAIL with import/module resolution errors because the package does not exist yet.

- [ ] **Step 3: Add minimal packaging and package entrypoints**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "openresearch"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Run test to verify imports resolve**

Run: `pytest tests/sandbox/test_local_docker_backend.py::test_package_imports -v`
Expected: PASS once the package exists.

### Task 2: Define runtime contracts and shared-volume conventions

**Files:**
- Create: `src/reprolab/sandbox/models.py`
- Create: `src/reprolab/sandbox/runtime_backend.py`
- Modify: `src/reprolab/sandbox/__init__.py`
- Test: `tests/sandbox/test_local_docker_backend.py`

- [ ] **Step 1: Write failing contract tests**

```python
async def test_create_sandbox_builds_expected_mount_layout() -> None:
    ...

def test_runtime_backend_contract_exports_expected_types() -> None:
    ...
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `pytest tests/sandbox/test_local_docker_backend.py -k "mount_layout or contract" -v`
Expected: FAIL because the config models and abstract backend are not implemented yet.

- [ ] **Step 3: Implement the shared models and abstract interface**

```python
@dataclass(frozen=True)
class SandboxConfig:
    image: str
    project_root: Path
    shared_volume_root: Path
    run_kind: RunKind
    run_id: str
    ...

class RuntimeBackend(ABC):
    @abstractmethod
    async def create_sandbox(self, config: SandboxConfig) -> Sandbox: ...
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `pytest tests/sandbox/test_local_docker_backend.py -k "mount_layout or contract" -v`
Expected: PASS for contract/type expectations while Docker lifecycle tests still fail.

### Task 3: Implement LocalDockerBackend lifecycle behavior

**Files:**
- Create: `src/reprolab/sandbox/local_docker_backend.py`
- Modify: `src/reprolab/sandbox/__init__.py`
- Test: `tests/sandbox/test_local_docker_backend.py`

- [ ] **Step 1: Write failing lifecycle tests with a fake Docker client**

```python
async def test_create_exec_copy_and_destroy_delegate_to_container_client() -> None:
    ...

def test_create_sandbox_uses_baseline_and_improvement_artifact_paths() -> None:
    ...
```

- [ ] **Step 2: Run targeted lifecycle tests to verify failure**

Run: `pytest tests/sandbox/test_local_docker_backend.py -k "delegate or artifact_paths" -v`
Expected: FAIL because `LocalDockerBackend` does not exist yet.

- [ ] **Step 3: Implement LocalDockerBackend**

```python
class LocalDockerBackend(RuntimeBackend):
    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        ...

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        ...

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        ...

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        ...

    async def destroy(self, sandbox: Sandbox) -> None:
        ...
```

- [ ] **Step 4: Run lifecycle tests to verify they pass**

Run: `pytest tests/sandbox/test_local_docker_backend.py -v`
Expected: PASS with Docker behavior covered by fake-client tests.

### Task 4: Verify the full MVP foundation and document gaps

**Files:**
- Modify: `tests/sandbox/test_local_docker_backend.py`
- Optional notes in final handoff only

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: PASS for all new sandbox tests.

- [ ] **Step 2: Attempt environment verification commands**

Run: `python -m compileall src`
Expected: PASS

Run: `docker --version`
Expected: may FAIL in this environment; if so, record that Docker-backed integration could not be exercised locally.

- [ ] **Step 3: Record implementation boundaries**

Document in the final handoff that:
- the repo did not contain dependency `#7` runtime scaffolding,
- the implementation therefore includes the missing abstraction foundation,
- Docker daemon/client integration is unit-tested through injection because Docker is unavailable in this workspace.
