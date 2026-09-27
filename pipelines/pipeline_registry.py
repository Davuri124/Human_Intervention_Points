"""
Pipeline Registry — Central hub for all pipeline types.
This module organizes all pipeline environments
and provides a clean API for the rest of the project.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments'))

from pipeline_env       import AIPipelineEnv
from pipeline_env_v2    import AIPipelineEnvV2
from pipeline_env_v3    import AIPipelineEnvV3
from multi_pipeline_env import MultiPipelineEnv, PIPELINE_CONFIGS
from cost_aware_env     import CostAwarePipelineEnv


# ══════════════════════════════════════════════════════════════
# PIPELINE REGISTRY
# One place to access all pipeline types
# ══════════════════════════════════════════════════════════════

REGISTRY = {
    "v1_baseline": {
        "class"      : AIPipelineEnv,
        "description": "V1 — Simulated random uncertainty",
        "version"    : 1,
        "obs_size"   : 5,
    },
    "v2_nonstationary": {
        "class"      : AIPipelineEnvV2,
        "description": "V2 — Non-stationary uncertainty with cascading errors",
        "version"    : 2,
        "obs_size"   : 8,
    },
    "v3_real_uncertainty": {
        "class"      : AIPipelineEnvV3,
        "description": "V3 — Real neural network uncertainty",
        "version"    : 3,
        "obs_size"   : 8,
    },
    "multi_pipeline": {
        "class"      : MultiPipelineEnv,
        "description": "Multi — 4 real-world domain pipelines",
        "version"    : 4,
        "obs_size"   : 12,
    },
    "cost_aware": {
        "class"      : CostAwarePipelineEnv,
        "description": "Cost — Expert-level cost-aware intervention",
        "version"    : 5,
        "obs_size"   : 14,
    },
}


def get_pipeline(name, config=None):
    """Get a pipeline environment by name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown pipeline: {name}. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name]["class"](config)


def list_pipelines():
    """Print all available pipelines."""
    print("\n📋 Available Pipelines:")
    print("=" * 55)
    for name, info in REGISTRY.items():
        print(f"  {name:<22} v{info['version']} — {info['description']}")
    print("=" * 55)


def get_domain_configs():
    """Return all real-world domain configurations."""
    return PIPELINE_CONFIGS