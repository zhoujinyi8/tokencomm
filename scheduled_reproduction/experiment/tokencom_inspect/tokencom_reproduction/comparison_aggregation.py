"""Complete-cohort paired aggregation for the registered ten-image comparison."""
import math

PIPELINES = ('demo', 'training_code_convention')
FIELDS = ('correct_tokens', 'whole_image_ter', 'psnr_raw', 'lpips_normalized', 'clip_clipped')


def summarize_complete_comparison(cases, paths):
    expected = {(pipeline, image, trial) for pipeline in PIPELINES for image in range(10) for trial in range(5)}
    keyed = {}
    for case in cases:
        key = (case['pipeline'], case['image_index'], case['trial'])
        if key not in expected or key in keyed:
            raise ValueError('Unexpected or duplicate pipeline/image/trial coordinate')
        if [row['path'] for row in case['rows']] != list(paths):
            raise ValueError('All planned paths must occur exactly once and in planned order')
        if any(not math.isfinite(row[field]) for row in case['rows'] for field in FIELDS):
            raise ValueError('Nonfinite metric')
        keyed[key] = {row['path']: row for row in case['rows']}
    if set(keyed) != expected:
        raise ValueError('Quality means require all 100 pipeline receptions and 50 matched pairs')
    per_image = [{pipeline: {name: {field: sum(keyed[pipeline, image, trial][name][field]
        for trial in range(5))/5 for field in FIELDS} for name in paths} for pipeline in PIPELINES} for image in range(10)]
    means = {pipeline: {name: {field: sum(item[pipeline][name][field] for item in per_image)/10
        for field in FIELDS} for name in paths} for pipeline in PIPELINES}
    differences = [{name: {field: item['training_code_convention'][name][field]-item['demo'][name][field]
        for field in FIELDS} for name in paths} for item in per_image]
    paired_means = {name: {field: sum(item[name][field] for item in differences)/10 for field in FIELDS} for name in paths}
    return {'means_over_images': means, 'per_image': per_image,
            'paired_image_differences': differences, 'paired_mean_difference': paired_means}
