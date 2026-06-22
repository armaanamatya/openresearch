

from backend.services.events.live_runs import StartRunRequest


def test_start_run_request_accepts_repo_url():
    req = StartRunRequest(repo_url="https://github.com/me/mine")
    assert req.repo_url == "https://github.com/me/mine"


def test_start_run_request_repo_url_defaults_none():
    assert StartRunRequest().repo_url is None


def test_start_arxiv_request_accepts_repo_url():
    from backend.app import StartArxivRunRequest
    req = StartArxivRunRequest(url="https://arxiv.org/abs/2605.15155", repo_url="github:me/mine")
    assert req.repo_url == "github:me/mine"


def test_cli_parses_repo_url():
    from backend.cli import _build_parser  # the argparse factory (confirmed symbol)
    parser = _build_parser()
    ns = parser.parse_args(["reproduce", "2605.15155", "--repo-url", "github:me/mine"])
    assert ns.repo_url == "github:me/mine"


def test_python_script_threads_repo_url(tmp_path):
    from backend.services.events.live_runs import _python_script
    req = StartRunRequest(repo_url="github:me/mine")
    script = _python_script(req, project_id="prj_x", runs_root=tmp_path, uploaded_paper=None)
    # The serialized config embedded in the subprocess script carries repo_url.
    assert "repo_url" in script
    assert "github:me/mine" in script
