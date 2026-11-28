"""Aggregate a fixed/random SNR pair only after both complete coordinate grids exist."""
import math

PATHS=('direct','without_cmi','with_cmi','ideal_correct_tokens')
KEYS=('correct_tokens','whole_image_ter','psnr_raw','lpips_normalized','clip_clipped')


def paired_snr_summary(fixed,randomized):
    expected={(i,t) for i in range(10) for t in range(5)}
    mapped=[]
    for cohort in (fixed,randomized):
        assert len(cohort)==50
        assert {(c['image_index'],c['trial']) for c in cohort}==expected
        assert len({c['snr_db'] for c in cohort})==1
        index={}
        for case in cohort:
            assert [row['path'] for row in case['rows']]==list(PATHS)
            assert all(math.isfinite(row[key]) for row in case['rows'] for key in KEYS)
            index[case['image_index'],case['trial']]={row['path']:row for row in case['rows']}
        mapped.append(index)
    assert fixed[0]['snr_db']==randomized[0]['snr_db']
    image_means=[]
    for index in mapped:
        image_means.append([{path:{key:sum(index[i,t][path][key] for t in range(5))/5
                                  for key in KEYS} for path in PATHS} for i in range(10)])
    differences=[{path:{key:image_means[1][i][path][key]-image_means[0][i][path][key]
                        for key in KEYS} for path in PATHS} for i in range(10)]
    return dict(receptions_per_stream=50,images=10,
                means_over_images={label:{path:{key:sum(image_means[j][i][path][key] for i in range(10))/10
                                                for key in KEYS} for path in PATHS}
                                   for j,label in enumerate(('fixed','randomized'))},
                per_image_trial_means=dict(fixed=image_means[0],randomized=image_means[1]),
                paired_image_differences=differences,
                paired_mean_difference={path:{key:sum(row[path][key] for row in differences)/10
                                               for key in KEYS} for path in PATHS})
