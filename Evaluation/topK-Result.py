import os


def getInformation(file):
    epoch = []
    training_accuracy = []
    training_loss = []
    test_accuracy = []
    test_fscore = []

    summarized_list = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for l in range(len(lines)):
            _line_information = lines[l].lower()

            if _line_information.startswith('epoch:'):
                splitted_info = _line_information.split(',')
                _eps = int(splitted_info[0][splitted_info[0].find('[') + 1:splitted_info[0].rfind(']')])
                _tr_acc = float(splitted_info[1][splitted_info[1].find('[') + 1:splitted_info[1].rfind(']')])
                _tr_loss = float(splitted_info[2][splitted_info[2].find('[') + 1:splitted_info[2].rfind(']')])
                epoch.append(_eps)
                training_accuracy.append(_tr_acc)
                training_loss.append(_tr_loss)

            if _line_information.startswith('mode: source'):
                _tst_accuracy_info = lines[l + 1].lower()
                _tst_fscore_info = lines[l + 9].lower()
                _source_tst_accuracy = float(_tst_accuracy_info.split(':')[1][1:])
                _source_tst_fscore = float(_tst_fscore_info.split(':')[1][1:])
                test_accuracy.append(_source_tst_accuracy)
                test_fscore.append(_source_tst_fscore)

    for e, tr_acc, tr_loss, tst_acc, tst_fsc in zip(epoch, training_accuracy, training_loss, test_accuracy,
                                                    test_fscore):
        summarized_list.append([e, tr_acc, tr_loss, tst_acc, tst_fsc])

    return summarized_list


def sortList(summarized_list, index):
    sorted_list = sorted(summarized_list, key=lambda s: s[index], reverse=True)

    return sorted_list


def showResults(sorted_list, count):
    for i in range(count):
        print(
            'Epoch [%d], Training Accuracy [%.4f], Test Accuracy [%.4f], Test Fscore [%.4f]' %
            (sorted_list[i][0], sorted_list[i][1], sorted_list[i][3], sorted_list[i][4]))
        # print()


def main():
    top_k = 20
    file_dir = 'D:/Workspace/Res2Net_CIFAR100/Result/'
    file_name_list = os.listdir(file_dir)
    print(file_name_list)
    for file in file_name_list:
        print('File Name {}'.format(file))
        info_list = getInformation(file_dir + file)
        sort_list = sortList(info_list, index=3)
        showResults(sort_list, top_k)


if __name__ == '__main__':
    main()
