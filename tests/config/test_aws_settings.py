"""AWS EKS foundation settings remain inert and opt-in by default."""

from backend.config import Settings


def test_aws_settings_defaults_do_not_select_or_configure_aws():
    settings = Settings(_env_file=None)

    assert settings.default_sandbox == "local"
    assert settings.aws_region == "us-east-1"
    assert settings.aws_eks_cluster == ""
    assert settings.aws_s3_bucket == ""
    assert settings.aws_base_image == ""
    assert settings.aws_gpu_skus == []
    assert settings.aws_max_nodes == 0
    assert settings.aws_gpus_per_node == 0
    assert settings.aws_per_gpu_vram_gb == 0.0
    assert settings.aws_gpu_usd_per_hour == 0.0


def test_aws_is_accepted_only_as_an_explicit_sandbox_choice():
    assert Settings(_env_file=None, default_sandbox="aws").default_sandbox == "aws"
    assert Settings(_env_file=None, force_sandbox="aws").force_sandbox == "aws"
