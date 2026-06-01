"""GEPA per-run sub-agent prompt optimization.

Entry points: gepa_pre_call, gepa_post_call (imported by binding.wrap_primitive).
All functions are no-ops when Settings().gepa_optimization == "off".
"""
