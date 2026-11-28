"""Small complete/incomplete cohort checks; no datasets or model inference."""
import copy
import math

from comparison_aggregation import summarize_complete_comparison

PATHS = ['direct', 'without_cmi', 'with_cmi', 'ideal_correct_tokens']


def fixture():
    cases = []
    for pipeline, delta in [('demo', 0), ('training_code_convention', 1)]:
        for image in range(10):
            for trial in range(5):
                rows = [{'path': path, 'correct_tokens': 250+delta, 'whole_image_ter': 1-(250+delta)/256,
                         'psnr_raw': 20+image+.1*trial+2*delta,
                         'lpips_normalized': .2+.001*image-.01*delta,
                         'clip_clipped': .8+.001*image+.02*delta} for path in PATHS]
                cases.append({'pipeline': pipeline, 'image_index': image, 'trial': trial, 'rows': rows})
    return cases


def expect_rejection(cases):
    try:
        summarize_complete_comparison(cases, PATHS)
    except ValueError:
        return
    raise AssertionError('Invalid cohort unexpectedly produced quality means')


def test_complete_pairing_and_order_independence():
    result = summarize_complete_comparison(list(reversed(fixture())), PATHS)
    assert math.isclose(result['means_over_images']['demo']['direct']['psnr_raw'], 24.7)
    assert math.isclose(result['means_over_images']['training_code_convention']['direct']['psnr_raw'], 26.7)
    assert all(math.isclose(row['direct']['psnr_raw'], 2) for row in result['paired_image_differences'])
    assert math.isclose(result['paired_mean_difference']['direct']['lpips_normalized'], -.01)


def test_incomplete_pair_rejected():
    expect_rejection(fixture()[:-1])


def test_duplicate_cannot_hide_missing_pair():
    cases = fixture(); cases[-1] = copy.deepcopy(cases[-2]); expect_rejection(cases)


def test_missing_path_rejected():
    cases = fixture(); cases[0]['rows'].pop(); expect_rejection(cases)


if __name__ == '__main__':
    checks = [test_complete_pairing_and_order_independence, test_incomplete_pair_rejected,
              test_duplicate_cannot_hide_missing_pair, test_missing_path_rejected]
    for check in checks:
        check(); print('PASS', check.__name__)
    print('4 paired-aggregation contract checks passed')
