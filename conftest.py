"""Declare tests requiring local experiment assets; preserve their assertions."""
import pytest

INTEGRATION_CASES = {
    "test_context_maskgit_v4.py::ContextMaskGITv4Tests::test_v3_warm_start_preserves_conditional_and_unconditional_logits",
    "test_goal_5hour.py::GoalFiveHourTests::test_status_generator_and_hidden_dispatch",
    "test_stage13.py::Stage13Tests::test_frozen_data_hashes_and_budget",
    "test_stage19.py::Stage19Tests::test_frozen_implementation_and_assets_match",
    "test_stage19.py::Stage19Tests::test_v4_checkpoint_and_condition_grid",
    "test_stage21_clip_context.py::Stage21ClipContextTests::test_generated_embeddings_align_with_food101_labels",
    "test_stage21_token_cache.py::Stage21TokenCacheTests::test_aggregate_is_fail_closed_while_cache_is_partial",
    "test_stage21_token_cache.py::Stage21TokenCacheTests::test_frozen_protocol_and_first_real_shard",
    "test_stage5.py::Stage5Artifacts::test_new_split_is_disjoint_and_test_locked",
    "test_stage5.py::Stage5Artifacts::test_training_runs_completed_without_test",
    "test_stage7.py::Stage7DataProtocol::test_third_split_is_disjoint_and_test_is_unopened",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in INTEGRATION_CASES:
            item.add_marker(pytest.mark.integration)
