"""Tests for backend.agents.rlm.requirements_derive."""

from __future__ import annotations

from pathlib import Path


from backend.agents.rlm import requirements_derive as rd


# ---------------------------------------------------------------------------
# parse_pip_packages_from_dockerfile
# ---------------------------------------------------------------------------


def test_parses_simple_single_run() -> None:
    df = """\
FROM python:3.11-slim
RUN pip install torch==2.2.0 numpy==1.26.4
WORKDIR /workspace
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["numpy==1.26.4", "torch==2.2.0"]


def test_parses_index_url_dropped() -> None:
    df = """\
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \\
    --index-url https://download.pytorch.org/whl/cpu
"""
    pkgs = rd.parse_pip_packages_from_dockerfile(df)
    assert pkgs == ["torch==2.2.0", "torchvision==0.17.0"]


def test_parses_multiline_continuation() -> None:
    df = """\
RUN pip install --no-cache-dir \\
    matplotlib==3.8.0 \\
    numpy==1.26.4 \\
    tqdm==4.66.0
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "matplotlib==3.8.0", "numpy==1.26.4", "tqdm==4.66.0",
    ]


def test_parses_chained_pip_install_blocks() -> None:
    df = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0
RUN pip install --no-cache-dir matplotlib==3.8.0 numpy==1.26.4
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "matplotlib==3.8.0", "numpy==1.26.4", "torch==2.2.0", "torchvision==0.17.0",
    ]


def test_drops_chained_shell_commands() -> None:
    df = """\
RUN pip install --no-cache-dir torch==2.2.0 && rm -rf /var/lib/apt/lists/*
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["torch==2.2.0"]


def test_dedupes_across_blocks() -> None:
    df = """\
RUN pip install torch==2.2.0
RUN pip install torch==2.2.0 numpy==1.26.4
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["numpy==1.26.4", "torch==2.2.0"]


def test_unversioned_specs_pass() -> None:
    df = "RUN pip install matplotlib numpy\n"
    assert rd.parse_pip_packages_from_dockerfile(df) == ["matplotlib", "numpy"]


def test_no_pip_install_returns_empty() -> None:
    df = "FROM python:3.11-slim\nWORKDIR /workspace\n"
    assert rd.parse_pip_packages_from_dockerfile(df) == []


def test_skips_git_and_url_specs() -> None:
    df = """\
RUN pip install torch==2.2.0 git+https://github.com/foo/bar.git
"""
    # git+ form belongs in Dockerfile only; not safe for requirements.txt.
    assert rd.parse_pip_packages_from_dockerfile(df) == ["torch==2.2.0"]


# ---------------------------------------------------------------------------
# synthesize_requirements_txt
# ---------------------------------------------------------------------------


def test_synthesize_includes_header_and_sorted_packages() -> None:
    df = "RUN pip install torch==2.2.0 matplotlib==3.8.0\n"
    out = rd.synthesize_requirements_txt(df)
    assert out.startswith("# Auto-derived from Dockerfile")
    lines = [l for l in out.splitlines() if l and not l.startswith("#")]
    assert lines == ["matplotlib==3.8.0", "torch==2.2.0"]


def test_synthesize_empty_when_no_packages() -> None:
    assert rd.synthesize_requirements_txt("FROM python:3.11\n") == ""


# ---------------------------------------------------------------------------
# ensure_requirements_txt (integration)
# ---------------------------------------------------------------------------


def test_ensure_skips_when_requirements_exists(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    pinned = "preexisting==1.0\n"
    (code_dir / "requirements.txt").write_text(pinned)
    (tmp_path / "Dockerfile").write_text("RUN pip install other==2.0\n")

    out = rd.ensure_requirements_txt(code_dir)
    assert out == code_dir / "requirements.txt"
    # Must NOT overwrite the pre-existing requirements.
    assert (code_dir / "requirements.txt").read_text() == pinned


def test_ensure_synthesizes_when_requirements_missing(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN pip install --no-cache-dir torch==2.2.0 \\\n"
        "    --index-url https://download.pytorch.org/whl/cpu\n"
        "RUN pip install --no-cache-dir matplotlib==3.8.0 numpy==1.26.4 tqdm==4.66.0\n"
    )

    out = rd.ensure_requirements_txt(code_dir)
    assert out == code_dir / "requirements.txt"
    content = (code_dir / "requirements.txt").read_text()
    assert "torch==2.2.0" in content
    assert "matplotlib==3.8.0" in content
    assert "numpy==1.26.4" in content
    assert "tqdm==4.66.0" in content
    # index-url MUST be stripped
    assert "--index-url" not in content
    assert "https://" not in content


def test_ensure_returns_none_when_no_dockerfile(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    assert rd.ensure_requirements_txt(code_dir) is None


def test_ensure_returns_none_when_dockerfile_has_no_pip(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\nCMD python\n")
    assert rd.ensure_requirements_txt(code_dir) is None


def test_ensure_writes_deterministic_sorted_output(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (tmp_path / "Dockerfile").write_text(
        "RUN pip install zebra==1.0 alpha==2.0 mango==3.0\n"
    )
    rd.ensure_requirements_txt(code_dir)
    content = (code_dir / "requirements.txt").read_text()
    pkg_lines = [l for l in content.splitlines() if l and not l.startswith("#")]
    assert pkg_lines == ["alpha==2.0", "mango==3.0", "zebra==1.0"]


# ---------------------------------------------------------------------------
# Dockerfile heredoc form (Lane D)
# ---------------------------------------------------------------------------


def test_parses_heredoc_single_block_multiple_pip_lines() -> None:
    df = """\
FROM python:3.11-slim
RUN <<EOF
pip install torch==2.2.0 numpy==1.26.4
pip install --no-cache-dir matplotlib==3.8.0
EOF
WORKDIR /workspace
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "matplotlib==3.8.0", "numpy==1.26.4", "torch==2.2.0",
    ]


def test_parses_heredoc_with_double_quoted_tag() -> None:
    df = """\
RUN <<"EOF"
pip install scipy==1.11.0 pandas==2.0.0
EOF
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "pandas==2.0.0", "scipy==1.11.0",
    ]


def test_parses_heredoc_with_single_quoted_tag() -> None:
    df = """\
RUN <<'EOF'
pip install scikit-learn==1.4.0
EOF
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["scikit-learn==1.4.0"]


def test_skips_heredoc_with_redirection_target() -> None:
    # Heredoc body is written to a file — its ``pip install`` lines are data,
    # not shell, and must NOT be parsed.
    df = """\
RUN <<EOF cat > /etc/pip.conf
pip install evil==1.0
EOF
RUN pip install legit==2.0
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["legit==2.0"]


def test_skips_heredoc_with_trailing_redirection_operator() -> None:
    df = """\
RUN <<EOF > /tmp/script.sh
pip install also-evil==1.0
EOF
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == []


def test_parses_heredoc_mixed_with_single_line_run_dedups() -> None:
    df = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.2.0
RUN <<EOF
pip install torch==2.2.0
pip install matplotlib==3.8.0 numpy==1.26.4
EOF
RUN pip install --no-cache-dir tqdm==4.66.0
"""
    # torch==2.2.0 appears in BOTH a single-line RUN and the heredoc — must
    # appear exactly once in the output.
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "matplotlib==3.8.0", "numpy==1.26.4", "torch==2.2.0", "tqdm==4.66.0",
    ]


def test_parses_heredoc_with_line_continuations_inside() -> None:
    df = """\
RUN <<EOF
pip install --no-cache-dir \\
    torch==2.2.0 \\
    numpy==1.26.4 \\
    --index-url https://download.pytorch.org/whl/cpu
EOF
"""
    pkgs = rd.parse_pip_packages_from_dockerfile(df)
    assert pkgs == ["numpy==1.26.4", "torch==2.2.0"]
    # index-url URL must be stripped, never leaked as a package.
    assert "https://download.pytorch.org/whl/cpu" not in pkgs


def test_parses_pip_inside_heredoc_conditional() -> None:
    # Modern LLM-generated Dockerfiles sometimes wrap pip in shell
    # conditionals inside heredocs — the substring match should still find
    # them.
    df = """\
RUN <<EOF
set -euo pipefail
if [ ! -f /tmp/installed ]; then
    pip install requests==2.31.0
fi
EOF
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == ["requests==2.31.0"]


def test_parses_multiple_heredoc_blocks() -> None:
    df = """\
RUN <<EOF
pip install alpha==1.0
EOF
RUN apt-get update && apt-get install -y git
RUN <<EOF
pip install beta==2.0 gamma==3.0
EOF
"""
    assert rd.parse_pip_packages_from_dockerfile(df) == [
        "alpha==1.0", "beta==2.0", "gamma==3.0",
    ]


def test_ensure_synthesizes_from_heredoc_dockerfile(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN <<EOF\n"
        "pip install --no-cache-dir torch==2.2.0 numpy==1.26.4\n"
        "pip install matplotlib==3.8.0\n"
        "EOF\n"
    )
    out = rd.ensure_requirements_txt(code_dir)
    assert out == code_dir / "requirements.txt"
    content = (code_dir / "requirements.txt").read_text()
    pkg_lines = [l for l in content.splitlines() if l and not l.startswith("#")]
    assert pkg_lines == ["matplotlib==3.8.0", "numpy==1.26.4", "torch==2.2.0"]
