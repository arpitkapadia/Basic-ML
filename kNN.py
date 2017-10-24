
import os
import struct
import numpy as np
import math
import random

"""
function to read dataset, used from below location
https://gist.github.com/akesling/5358964
"""

def read(dataset = "training", path = ".", return_generator = True, index = -1):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    if(not return_generator):
        yield get_img(index)

    # Create an iterator which returns each image in turn
    if(return_generator):
        for i in range(len(lbl)):
            yield get_img(i)

def calculate_euclidian_distance(image1, image2):
    distance = 0
    for i in range(28):
        for j in range(28):
            distance += pow((int(image1[i, j]) - int(image2[i, j])), 2)
    return math.sqrt(distance)

def getFractionOfTrainingSamples(length_of_training_data, fraction):
    index_set = set()
    sample_data_length = int(length_of_training_data * fraction)
    while(len(index_set) < sample_data_length):
        index = random.randrange(length_of_training_data)
        index_set.add(index)
    return index_set

def generate_distance_matrix_testing_image(image, training_data_indexes, path = "/Users/ak/Downloads/575/asgn2"):
    return_list = []
    for index in training_data_indexes:
        image_data = read("training", path, False, index)
        training_label, training_image = next(image_data)

        return_list.append((training_label, index, calculate_euclidian_distance(image, training_image)))
    return return_list

def classify_summarize_test_images(nearest_neighbours = 5, path = "/Users/ak/Downloads/575/asgn2", length_of_training_data = 60000, fraction = 0.05):
    training_index_set = getFractionOfTrainingSamples(length_of_training_data, fraction)
    print("training length " + str(len(training_index_set)))

    test_image_results = []
    postive_prediction = 0
    negative_prediction = 0

    for test_label, test_image in read("testing", path):
        distance_list = generate_distance_matrix_testing_image(test_image, training_index_set)
        distance_list = sorted(distance_list, key = lambda x : x[2], reverse = True)
        label_count_list = [0]*10
        positive_prediction = 0
        negative_prediction = 0
        for neighbour in range(nearest_neighbours):
            label_count_list[distance_list[neighbour][0]] += 1
        predicted_class = label_count_list.index(max(label_count_list))
        if(predicted_class == test_label):
            positive_prediction += 1
        else:
            negative_prediction += 1
    return (positive_prediction, negative_prediction)

def summarize_based_on_nearest_neighbours(neighbour_count_list, path = "/Users/ak/Downloads/575/asgn2", length_of_training_data = 60000, fraction = 0.001):
    accuracy_list = []
    for value in neighbour_count_list:
        positive, negative = classify_summarize_test_images(value, path, length_of_training_data, fraction)
        accuracy_list.append(positive/(positive + negative))

    plot_graph(neighbour_count_list, accuracy_list)

def plot_graph(list1, list2):

    plt.plot(list1, list2, 'ro')
    # plt.axis([0, 6, 0, 20])
    plt.show()

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    print("inshow")
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

if __name__ == '__main__':
    summarize_based_on_nearest_neighbours([1,3])
    #[1,3,5,10,30,50,70,80,90,100]
    # mnist = read("training", "/Users/ak/Downloads/575/asgn2")
    # image = mnist.next()
    # print("label of image 1 is " + str(image[0]))
    # #show(image[1])
    #
    # image1 = mnist.next()
    # print("label of image 2 is " + str(image1[0]))
    # #show(image1[1])
    # print("euclidian distance is " + str(calculate_euclidian_distance(image[1], image1[1])))
    # show(next(read("training", "/Users/ak/Downloads/575/asgn2")))
