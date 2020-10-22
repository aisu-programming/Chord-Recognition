import mir_eval

def get_sevenths_scores_array(ref_file, est_file):

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

    # durations = mir_eval.util.intervals_to_durations(intervals)
    comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
    # score = mir_eval.chord.weighted_accuracy(comparisons, durations)

    # print(score)
    return comparisons

# for testing
if __name__ == "__main__":
    comparisons = get_sevenths_scores_array(
        ref_file='testing/ref_file.txt',
        est_file='testing/est_file.txt'
    )
    print(comparisons)