''' Libraries '''
import mir_eval

from mapping import my_mapping_dict


''' Parameters '''
REFERENCE_ERROR_REPAIR = True
REFERENCE_ERROR_REPAIR_POINT = 1


''' Codes '''
def get_sevenths_score(ref_file, est_file):

    (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(ref_file)
    (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_file)

    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals,
        est_labels,
        ref_intervals.min(),
        ref_intervals.max(),
        mir_eval.chord.NO_CHORD,
        mir_eval.chord.NO_CHORD
    )
    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals,
        ref_labels,
        est_intervals,
        est_labels
    )

    # for index, interval in enumerate(intervals):
    #     print(f'{interval[0]:0>10.6f} ~ {interval[1]:0>10.6f}: {ref_labels[index]:8} <---> {est_labels[index]:8}')

    durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)

    if REFERENCE_ERROR_REPAIR:
        for index in range(len(comparisons)):
            if comparisons[index] == -1:
                comparisons[index] = REFERENCE_ERROR_REPAIR_POINT

    # for index, comparison in enumerate(comparisons):
    #     if ('(' in ref_labels[index] or '/' in ref_labels[index]) and comparison != 0:
    #         print(f'ref_label <---> est_label:   {ref_labels[index]:10} <---> {est_labels[index]:10}: {comparison} * {durations[index]}')

    score = mir_eval.chord.weighted_accuracy(comparisons, durations)
    return score


''' Testing '''
if __name__ == "__main__":

    average_score = 0

    for i in range(200):

        ref_file = f'CE200/{i+1}/ground_truth.txt'
        est_file = f'CE200/{i+1}/est_file.txt'

        score = get_sevenths_score(ref_file=ref_file, est_file=est_file)
        print(f'\nReference file:\t\t{ref_file}\nEstimatation file:\t{est_file}\n\nScore: {score}')

        average_score += score

    print(average_score / 200)