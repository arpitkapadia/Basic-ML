import numpy as numpy
output_matrix = numpy.random.random((3,3))
global_word_list = ["bc", "mc", "cc"]
with open("/Users/ak/Downloads/575/asgn2/word_matrix_data_temp.txt",'w') as matrix_file:
    # output_matrix.tofile(matrix_file, sep=" ", format = "%s")
    i = 0
    # for line in output_matrix:
        # np.savetxt(matrix_file, x, '%.2f')
    # matrix_file.write("\n"+join("      ".join(map(str, x))
    for x in output_matrix:
        row = " ".join(map('{0:3f}'.format, x))
        matrix_file.write('{0:15} ==> '.format(global_word_list[i]) + row + "\n")

        # matrix_file.write(global_word_list[i] + " " +row)
        i += 1
