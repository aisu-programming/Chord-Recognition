import numpy as np

def dp_post_processing(input, alpha=0.25):  # alpha > 0
    frame_number = len(input[0])
    cost_list = []
    output = []
    cost_list_element = [float("-inf"), 0]

    for i in range(22):
        cost_list.append([])
        output.append([])
    for i in range(len(input[0])):  # Frame number
        for j in range(13):
            cost_list_element = [float("-inf"), 0]
            if i == 0:
                cost_list_element[0] = np.log(input[j][0])
            else:
                for k in range(13):
                    if j == k:  # No need to minus alpha
                        if cost_list[k][i - 1][0] + np.log(input[j][i]) > cost_list_element[0]:
                            cost_list_element[0] = cost_list[k][i - 1][0] + np.log(input[j][i])
                            cost_list_element[1] = k
                    else: 
                        if cost_list[k][i - 1][0] + np.log(input[j][i]) - alpha > cost_list_element[0]:
                            cost_list_element[0] = cost_list[k][i - 1][0] + np.log(input[j][i]) - alpha
                            cost_list_element[1] = k
            cost_list[j].append(cost_list_element)
        for j in range(13, 22):
            cost_list_element = [float("-inf"), 0]
            if i == 0:
                cost_list_element[0] = np.log(input[j][0])
            else:
                for k in range(13, 22):
                    if j == k:  # No need to minus alpha
                        if cost_list[k][i - 1][0] + np.log(input[j][i]) > cost_list_element[0]:
                            cost_list_element[0] = cost_list[k][i - 1][0] + np.log(input[j][i])
                            cost_list_element[1] = k
                    else:
                        if cost_list[k][i - 1][0] + np.log(input[j][i]) - alpha > cost_list_element[0]:
                            cost_list_element[0] = cost_list[k][i - 1][0] + np.log(input[j][i]) - alpha
                            cost_list_element[1] = k
            cost_list[j].append(cost_list_element)
    max_root_cost = float("-inf")
    max_root = -1
    max_chord_cost = float("-inf")
    max_chord = -1
    chord_list = []
    for i in range(13):
        if cost_list[i][frame_number - 1][0] > max_root_cost:
            max_root_cost = cost_list[i][frame_number - 1][0]
            max_root = cost_list[i][frame_number - 1][1]
    for i in range(13, 22):
        if cost_list[i][frame_number - 1][0] > max_chord_cost:
            max_chord_cost = cost_list[i][frame_number - 1][0]
            max_chord = cost_list[i][frame_number - 1][1]
    for i in reversed(range(frame_number)):
        if max_root == 12:  # No chord
            chord_list.append([12, -1])
        else:
            chord_list.append([max_root, max_chord])
        max_root = cost_list[max_root][i][1]
        max_chord = cost_list[max_chord][i][1]
    for i in reversed(range(frame_number)):
        for j in range(13):
            if j == chord_list[i][0]:
                output[j].append(1.0)
            else:
                output[j].append(0.0)
        if chord_list[i][0] == 12:
            for j in range(13, 22):
                output[j].append(0.0)
        else:
            for j in range(13, 22):
                if j == chord_list[i][1]:
                    output[j].append(1.0)
                else:
                    output[j].append(0.0)
    return np.array(output)
