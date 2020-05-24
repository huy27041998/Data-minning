import random
import os
import numpy as np
class LoadData:
    def __init__(self):
        super().__init__()
    def loadData(self, limit, class_name=None):
        if class_name:
            with open('data/' + class_name, 'r', encoding='utf-8') as f:
                data = f.read().splitlines()
                f.close()
            return {'class_name' : class_name[:limit], 'data': random.choices(self.handleData(data), k=limit)}
        else:
            dirs = os.listdir('data/')
            result = {'class_name': [], 'data': []}
            for class_name in dirs:
                with open('data/' + class_name, 'r', encoding='utf-8') as f:
                    datas = f.read().splitlines()
                    temp_class_name = []
                    temp_data = []
                    for data in datas:
                        # result['class_name'].append(class_name)
                        # result['data'].append(data)
                        temp_class_name.append(class_name)
                        temp_data.append(data)
                    result['data'].extend(random.choices(temp_data, k = limit))
                    result['class_name'].extend(temp_class_name[:limit])
                    f.close()
            return result
    def loadStopWords(self):
        with open('stop-word', 'r', encoding='utf-8') as f:
            self.stopWords = f.read().splitlines()
        return self.stopWords
    def handleData(self, data):
        return data
