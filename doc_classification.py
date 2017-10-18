import numpy as np
def getwordmatrix(write_to_file):
    word_dict = {}
    global_word_set = set()
    list_of_dict = []
    with open("/Users/ak/Downloads/575/asgn2/adv_data.txt") as data_file:
        for line in data_file:
            word_dict = {}
            word_list = line.split()
            for word in word_list:
                word_dict.setdefault(word,0)
                word_dict[word] += 1
                global_word_set.add(word)
            list_of_dict.append(word_dict)

    # file matrix_file = open("/Users/ak/Downloads/575/asgn2/word_matrix_data.txt",r+)
    global_word_list = list(global_word_set)
    global_word_list.sort()
    output_matrix = np.zeros((len(global_word_list), len(list_of_dict)))

    i = 0
    j = 0
    print("number of words" + str(len(global_word_list)) + " " + str(len(list_of_dict)))

    for line_dict in list_of_dict:
        i=0
        for word in global_word_list:
            if word in line_dict:
                output_matrix[i,j] = line_dict[word]
            else:
                output_matrix[i,j] = 0
            i += 1
        j += 1

    if write_to_file:
        with open("/Users/ak/Downloads/575/asgn2/word_matrix_data.txt",'w') as matrix_file:
            i = 0
            for x in output_matrix:
                row = " ".join(map('{0:4.0f}'.format, x))
                matrix_file.write('{0:20}*'.format(global_word_list[i]) + row + "\n")
                i += 1

    return (global_word_list, output_matrix)
