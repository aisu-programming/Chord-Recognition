''' Libraries '''
import sys
import mir_eval

from mapping import my_mapping_dict


''' Codes '''
def get_sevenths_score(ref_file, est_file, error_repair=False, error_repair_point=0):

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

    if error_repair:
        for index in range(len(comparisons)):
            if comparisons[index] == -1:
                comparisons[index] = error_repair_point

    # for index, comparison in enumerate(comparisons):
    #     if ('(' in ref_labels[index] or '/' in ref_labels[index]) and comparison != 0:
    #         print(f'ref_label <---> est_label:   {ref_labels[index]:10} <---> {est_labels[index]:10}: {comparison} * {durations[index]}')

    score = mir_eval.chord.weighted_accuracy(comparisons, durations)
    return score


''' Testing '''
if __name__ == "__main__":
    
    output_details = False

    average_score = 0
    average_repair_0_score = 0
    average_repair_1_score = 0

    if not output_details:
        toolbar_width = 100
        sys.stdout.write("\n[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))
    for i in range(200):

        ref_file = f'CE200/{i+1}/ground_truth.txt'
        est_file = f'CE200/{i+1}/est_file.txt'

        if output_details: print(f'\n--------------------------------------------------\n')
        if output_details: print(f'Reference file:    {ref_file}\nEstimatation file: {est_file}\n')

        score = get_sevenths_score(ref_file=ref_file, est_file=est_file)
        average_score += score
        if output_details: print(f'Score not repaired:       {score}')

        score = get_sevenths_score(ref_file=ref_file, est_file=est_file, error_repair=True)
        average_repair_0_score += score
        if output_details: print(f'Score repaired (point=0): {score}')

        score = get_sevenths_score(ref_file=ref_file, est_file=est_file, error_repair=True, error_repair_point=1)
        average_repair_1_score += score
        if output_details: print(f'Score repaired (point=1): {score}')

        if not output_details and (i+1) % 2 == 0:
            sys.stdout.write("=")
            sys.stdout.flush()
            
    if not output_details: sys.stdout.write("]\n\n")

    if output_details: print(f'\n--------------------------------------------------\n')
    print(f'Average score not repaired         : {average_score / 200}')
    print(f'Average score repaired (point = 0) : {average_repair_0_score / 200}')
    print(f'Average score repaired (point = 1) : {average_repair_1_score / 200}')