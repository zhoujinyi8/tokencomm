"""Render all complete-cohort paired image deltas without ranking pipelines."""
import math


def plot_paired_differences(audit, outputs, prefix):
    assert audit['complete'] and audit['complete_pairs'] == 50
    differences = audit['paired_image_differences']
    assert len(differences) == 10
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = ['direct', 'without_cmi', 'with_cmi', 'ideal_correct_tokens']
    labels = ['Direct', 'Without\nCMI', 'With\nCMI', 'Ideal\ntokens']
    fields = [('psnr_raw', 'PSNR difference (dB)'), ('lpips_normalized', 'LPIPS difference'),
              ('clip_clipped', 'CLIP cosine difference')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, (field, ylabel) in zip(axes, fields):
        means = []
        for i, path in enumerate(paths):
            values = [row[path][field] for row in differences]
            assert all(math.isfinite(value) for value in values)
            mean = audit['paired_mean_difference'][path][field]
            assert math.isclose(sum(values)/10, mean, abs_tol=1e-12)
            axis.scatter([i+(j-4.5)*.027 for j in range(10)], values, color='#2166ac', alpha=.65,
                         s=28, label='Per-image delta (n=10)' if i == 0 else None)
            means.append(mean)
        axis.scatter(range(4), means, marker='D', color='black', s=42, label='Mean paired delta', zorder=4)
        axis.axhline(0, color='gray', linestyle='--', linewidth=1)
        axis.set_xticks(range(4), labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Old-development matched comparison: 10 images x 5 receptions\n'
                 'Training-code convention minus demo; no pipeline selected', fontsize=12)
    fig.supxlabel('Points average five receptions within image. Positive favors training-code mapping for PSNR/CLIP; negative for LPIPS.', fontsize=9)
    figures = []
    for suffix in ('svg', 'png'):
        path = outputs/f'{prefix}_配对差值.{suffix}'
        fig.savefig(path, dpi=160); figures.append(path)
    plt.close(fig)
    return figures
