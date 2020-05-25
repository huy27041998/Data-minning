class LoadData:
    def __init__(self):
        super().__init__()
    def loadData(self, filename):
        result = {'class_name': [], 'data': []}
        with open(filename, 'r', encoding='utf-8') as f:
            datas = f.read().splitlines()
            data_size = len(datas[0].split(',')[1:])
            data_dict = [{} for i in range(data_size)]
            j = [0 for i in range(data_size)]
        for data in datas:
            contents = data.split(',')
            temp = []
            for index, content in enumerate(contents):
                if not isfloat(content) and len(content) > 0:
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
                    temp.append(float(content))
            result['data'].append(temp)
        return result
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
