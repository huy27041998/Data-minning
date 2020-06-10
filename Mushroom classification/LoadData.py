import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
class LoadData:
    def __init__(self):
        super().__init__()
    def loadData(self, filename):
        result = {'class_name': [], 'data': []}
        with open(filename, 'r', encoding='utf-8') as f:
            datas = f.read().splitlines()
            class_name_dict = {}
            data_size = len(datas[0].split(',')[1:])
            data_dict = [{} for i in range(data_size)]
            i = 0
            j = [0 for i in range(data_size)]
        for data in datas:
            contents = data.split(',')
            class_name = contents[0]
            if not class_name.isnumeric():
                if class_name not in class_name_dict:
                    class_name_dict[class_name] = i
                    i += 1
                result['class_name'].append(class_name_dict[class_name])
            else:
                result['class_name'].append(int(class_name))
            temp = []
            for index, content in enumerate(contents[1:]):
                if not content.isnumeric() and len(content) > 0:
                    if content not in data_dict[index]:
                        # la chu
                        data_dict[index][content] = j[index]
                        j[index] += 1
                    temp.append(data_dict[index][content])
                elif len(content) == 0:
                    # Khuyet thuoc tinh
                    temp.append(None)
                else:
                    # la so
                    temp.append(int(content))
            result['data'].append(temp)
<<<<<<< HEAD
        return result
    def loadDataOneHot(self, filename):
        result = {'class_name': [], 'data': []}
        with open(filename, 'r', encoding='utf-8') as f:
            datas = f.read().splitlines()
        for data in datas:
            contents = data.split(',')
            class_name = contents[0]
            result['class_name'].append(class_name)
            temp = []
            for content in contents[1:]:
                if is_number(content):
                    temp.append(int(content))
                else:
                    temp.append(content)
            result['data'].append(temp)
        x = np.array(result['data'])
        # print(x.shape)
        # enc = OneHotEncoder(dtype=np.int, sparse=False)
        # y = enc.fit_transform(x)
        # print(y.shape)
        # print(y[:, 0])
        print(x.shape)
        for i in range(x.shape[1]):
            column = x[:, i]
            print(column.shape)
            if not all(is_number(i) for i in column):
                y = pd.get_dummies(column)
                print(y.to_numpy())
            # if all(i )
        return result
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
=======
        return result
>>>>>>> d4f24e18d98f36bd756041dfa27780af9dac94fc
