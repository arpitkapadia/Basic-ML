import random
import matplotlib.pyplot as plt
import math
class LogisticRegression:
    def __init__(self, data_file, label_file):
        self.data_file = data_file
        self.label_file = label_file
        self.positive_count = 0
        self.negative_count = 0
        self.training_ads = []
        self.training_data_fraction = []
        self.error_list = []
        self.coefficients={}
        self.coefficient_offset = 0.0

    def getLabelInfoFromFile(self):
        ad_label_dict = {}
        with open(self.label_file,'r') as labels:
            for line in labels:
                ad_no, label = map(int, line.split())
                ad_label_dict[ad_no-1] = label
        #print(str(ad_label_dict))
        return ad_label_dict

    def getTrainingTestingData(self, split_fraction):
        ad_label_dict = self.getLabelInfoFromFile()
        list_of_dict = self.getAdsFromFile()

        total_ads = len(list_of_dict)
        total_ads_in_training = int(total_ads * split_fraction)
        #print("total training adds " + str(total_ads_in_training))
        global_set = set([i for i in range(total_ads)])
        training_set = set()
        testing_set = set()
        counter = 0
        while(len(training_set) < total_ads_in_training):
            training_set.add(random.randrange(total_ads))
        #print("training " + str(training_set) )
        testing_set = global_set - training_set
        #print("testing " + str(testing_set) )


        training_list_of_ad = []
        for ad in training_set:
            training_list_of_ad.append((ad, ad_label_dict[ad], list_of_dict[ad]))
        testing_list_of_ad = []
        for ad in testing_set:
            testing_list_of_ad.append((ad, ad_label_dict[ad], list_of_dict[ad]))
        return (training_list_of_ad, testing_list_of_ad)


    def getAdsFromFile(self):
        word_dict = {}
        global_word_set = set()
        list_of_dict = []
        with open(self.data_file) as data_file:
            for line in data_file:
                word_dict = {}
                word_list = line.split()
                for word in word_list:
                    word_dict.setdefault(word,0)
                    word_dict[word] += 1
                    global_word_set.add(word)
                list_of_dict.append(word_dict)

        # self.summary_basedon_words = self.summary_basedon_words(global_word_set, list_of_dict)
        #print(str(list_of_dict))
        return list_of_dict

    def summarizeBasedOnWords(self, training_list_of_ad, fraction_of_training_data):
        # training_list_of_ad, testing_list_of_ad = self.getTrainingTestingDat
        total_len = len(training_list_of_ad) * fraction_of_training_data
        new_training_list = []
        while(len(new_training_list) < total_len):
            index = random.randrange(len(training_list_of_ad))
            new_training_list.append(training_list_of_ad[index])
        # accepted_ads = [x[2] for x in new_training_list if x[1] == 1]
        # rejected_ads = [x[2] for x in new_training_list if x[1] == 0]
        self.coefficients = {}
        self.training_ads = new_training_list
        for ad, label, ad_dict in new_training_list:
            for word, count in ad_dict.items():
                # word_count_dict.setdefault((word, count),0)
                self.coefficients[(word, count)] = 0
                # word_count_set.add((word, count))

        # positive_word_dict = {}
        # negative_word_dict = {}
        # self.positive_count = len(accepted_ads)
        # self.negative_count = len(rejected_ads)
        #
        # for ad_dict in accepted_ads:
        #     for word, count in ad_dict.items():
        #         self.training_word_set.add(word)
        #         positive_word_dict.setdefault((word, count),0)
        #         positive_word_dict[(word, count)] += 1
        #
        # for ad_dict in rejected_ads:
        #     for word, count in ad_dict.items():
        #         self.training_word_set.add(word)
        #         negative_word_dict.setdefault((word, count),0)
        #         negative_word_dict[(word, count)] += 1
        # #print(str(positive_word_dict) +"  " + str(negative_word_dict))

    def getCurrentPrediction(self, ad_dict):
        prediction = self.coefficient_offset

        for word, count in ad_dict.items():
            if((word, count) in self.coefficients):
                prediction += self.coefficients[(word, count)] * 1
        return 1.0/(1.0 + math.exp(-prediction))

    def runGradientDescent(self, number_of_times, learning_rate):
        for i in range(number_of_times):
            sum_of_error = 0
            for ad, label, ad_dict in self.training_ads:
                current_prediction = self.getCurrentPrediction(ad_dict)
                error = float(label) - current_prediction
                sum_of_error += error ** 2
                self.coefficient_offset = self.coefficient_offset + (learning_rate * error * current_prediction * (1-current_prediction))
                for word, count in ad_dict.items():
                    if((word, count) in self.coefficients):
                        self.coefficients[(word, count)] += (learning_rate * error * current_prediction * (1-current_prediction))
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, learning_rate, sum_of_error))

    def predictTheFutureOfAd(self, fraction):
        training_list_of_ad, testing_list_of_ad = self.getTrainingTestingData(fraction)
        for fraction_of_training_data in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:

            self.training_ads = []
            self.summarizeBasedOnWords(training_list_of_ad, fraction_of_training_data)
            print("features considered %d training data length %d testing data length %d"%(len(self.coefficients), len(self.training_ads), len(testing_list_of_ad)))
            self.runGradientDescent(100, 0.01)


            classification_error_count = 0
            total_test_ads = len(testing_list_of_ad)
            # training_data_fraction = []
            self.training_data_fraction.append(fraction_of_training_data)
            # error_list = []

            for ad, actual_class, item in [(x[0], x[1], x[2]) for x in testing_list_of_ad]:
                current_predication = round(self.getCurrentPrediction(item))

                if(not current_predication == actual_class):
                    classification_error_count += 1.0

            error = classification_error_count / total_test_ads

            print("error " + str(error))

            self.error_list.append(classification_error_count / total_test_ads)

    def plotTheGraph(self):
        print("training fraction " + str(self.training_data_fraction))
        print("error list " + str(self.error_list))
        plt.plot(self.training_data_fraction, self.error_list, 'ro')
        # plt.axis([0, 6, 0, 20])
        plt.show()



            #print("predicted_class: " + str(predicted_class) + " actual_class: " + str(actual_class) +" "+ str((prob_positive_class,prob_negative_class)) )


if __name__ == '__main__':
    nb1 = LogisticRegression("/Users/ak/Downloads/575/asgn2/adv_data.txt", "/Users/ak/Downloads/575/asgn2/adv_label.txt")
    print("main hello")

    nb1.predictTheFutureOfAd(0.7)
    nb1.plotTheGraph()
