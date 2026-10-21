"""Exploratory paired statistics; no model, image, dataset or network access."""

import math
from collections import Counter
from itertools import combinations

from .analysis_metrics import analyze_predictions


MODELS = ('a', 'b', 'd')


def wilson(correct, total):
    """Two-sided 95% score interval, including the boundary cases 0/n and n/n."""
    if (type(correct) is not int or type(total) is not int
            or not 0 <= correct <= total):
        raise ValueError('Wilson requires integer counts 0 <= correct <= total.')
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total
                         + z * z / (4 * total * total)) / denominator
    return {'confidence': .95, 'lower': max(0., center - half),
            'upper': min(1., center + half), 'method': 'Wilson score'}


def exact_mcnemar(first_only, second_only):
    """Exact two-sided conditional binomial test over discordant pairs."""
    if any(type(n) is not int or n < 0 for n in (first_only, second_only)):
        raise ValueError('McNemar requires nonnegative integer discordant counts.')
    discordant = first_only + second_only
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return {'discordant_pairs': discordant,
            'first_only_correct': first_only, 'second_only_correct': second_only,
            'p_value_two_sided': min(1., 2 * tail / (1 << discordant)),
            'method': 'exact conditional binomial, doubled smaller tail',
            'informative': discordant > 0,
            'interpretation': 'Exploratory; no multiplicity adjustment. Non-significance is not equivalence.'}


def compare_pair(rows, first, second):
    """Count and identify every transition, also when both models are wrong."""
    outcomes = {k: [] for k in ('both_correct', 'both_incorrect', 'first_only_correct',
                                'second_only_correct', 'first_lower_distance',
                                'second_lower_distance', 'equal_distance')}
    exported = []
    for row in rows:
        identity = row['split'] + '/' + row['filename']
        left, right = row['correct_' + first], row['correct_' + second]
        group = ('both_correct' if left and right else 'first_only_correct' if left else
                 'second_only_correct' if right else 'both_incorrect')
        delta = row['edit_distance_' + second] - row['edit_distance_' + first]
        distance_group = ('second_lower_distance' if delta < 0 else
                          'first_lower_distance' if delta > 0 else 'equal_distance')
        outcomes[group].append(identity)
        outcomes[distance_group].append(identity)
        exported.append({'filename': row['filename'], 'split': row['split'],
                         'expected': row['expected'], 'pair': first.upper() + '_vs_' + second.upper(),
                         'first_model': first.upper(), 'second_model': second.upper(),
                         'predicted_first': row['predicted_' + first],
                         'predicted_second': row['predicted_' + second],
                         'correct_first': left, 'correct_second': right,
                         'outcome': group, 'improvement_exact': not left and right,
                         'regression_exact': left and not right,
                         'edit_distance_first': row['edit_distance_' + first],
                         'edit_distance_second': row['edit_distance_' + second],
                         'delta_edit_distance_second_minus_first': delta,
                         'distance_outcome': distance_group})
    counts = {key: len(value) for key, value in outcomes.items()}
    assert sum(counts[k] for k in ('both_correct', 'both_incorrect', 'first_only_correct',
                                  'second_only_correct')) == len(rows)
    assert sum(counts[k] for k in ('first_lower_distance', 'second_lower_distance',
                                  'equal_distance')) == len(rows)
    return exported, {'first': first.upper(), 'second': second.upper(), 'n': len(rows),
                      'counts': counts, 'images': outcomes,
                      'mcnemar': exact_mcnemar(counts['first_only_correct'], counts['second_only_correct'])}


def summarize(rows):
    """Use the existing deterministic alignment and weighted CER for each scope."""
    result = {}
    for scope in ('train', 'validation', 'aggregate'):
        selected = [r for r in rows if scope == 'aggregate' or r['split'] == scope]
        metrics = {}
        for model in MODELS:
            if not selected:
                metrics[model] = None
                continue
            predictions = [{'filename': r['split'] + '/' + r['filename'],
                            'expected': r['expected'], 'predicted': r['predicted_' + model]}
                           for r in selected]
            _, stats = analyze_predictions(predictions)
            for part in (stats['global'], *stats['by_length'].values()):
                part['exact_match_wilson_95'] = wilson(part['num_correct'], part['num_images'])
            length_counts = Counter((r['expected_length'], r['predicted_length_' + model]) for r in selected)
            stats['predicted_length_frequencies'] = [
                {'expected_length': expected, 'predicted_length': predicted,
                 'num_images': count, 'incorrect_length': expected != predicted}
                for (expected, predicted), count in sorted(length_counts.items())]
            metrics[model] = stats
        result[scope] = {'models': metrics, 'paired': {
            left.upper() + '_vs_' + right.upper(): compare_pair(selected, left, right)[1]
            for left, right in combinations(MODELS, 2)}}
    return result
