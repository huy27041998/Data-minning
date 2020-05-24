class LoadData:
    def __init__(self):
        super().__init__()
    def loadData(self):
        result = {'class_name': [], 'data': []}
        with open('mushrooms.csv', 'r', encoding='utf-8') as f:
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
        return result
l = LoadData()
l.loadData()