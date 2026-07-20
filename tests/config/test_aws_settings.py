"""AWS EKS foundation settings remain inert and opt-in by default."""

from backend.config import Settings


def test_aws_settings_defaults_do_not_select_or_configure_aws():
    settings = Settings(_env_file=None)

    assert settings.default_sandbox == "runpod"
    assert settings.aws_region == "us-east-1"
    assert settings.aws_eks_cluster == ""
    assert settings.aws_s3_bucket == ""
    assert settings.aws_base_image == ""
    assert settings.aws_gpu_skus == []


def test_aws_is_accepted_only_as_an_explicit_sandbox_choice():
    assert Settings(_env_file=None, default_sandbox="aws").default_sandbox == "aws"
    assert Settings(_env_file=None, force_sandbox="aws").force_sandbox == "aws"
