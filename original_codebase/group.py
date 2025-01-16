from collections import defaultdict

def group_train_data(train_data, train_labels, num):
    grouped_train_data = defaultdict(list)

    for data,label in zip(train_data, train_labels):
        grouped_train_data[label].append(data)
    
    grouped_data = {}
    for label, data in grouped_train_data.items():
        grouped_data[label] = [data[i:i+num] for i in range(0, len(data), num)]
    print("Grouped data: ", grouped_data)
    return grouped_data
    
