import numpy as np

f = open('./list_eval_partition.txt', 'r')
data = f.readlines()
data_list_train = list()
data_list_query = list()
data_list_gallery = list()

for line in data[2:]:
    #x = line.split("\n")
    #x = line.strip()
    x = line.strip().split(' ')

    file_path = x[0]
    #print('file path: ' + file_path)

    class_id_str = x[-2]
    class_id = int(class_id_str.split('_')[-1])
    #print('class_id: ' + str(class_id))

    partition_id = x[-1]
    #print('partition: ' + partition_id)
    if partition_id == 'train':
        data_list_train.append(file_path+' '+str(class_id)+'\n' )

    if partition_id == 'gallery':
        data_list_gallery.append(file_path+' '+str(class_id)+'\n' )

    if partition_id == 'query':
        data_list_query.append(file_path+' '+str(class_id)+'\n' )



print(data_list_train[0])
print("Train Size:", len(data_list_train))
print("Query Size:", len(data_list_query))
print("Gallery Size:", len(data_list_gallery))

train_file = open('train.txt', 'w')
train_file.writelines(data_list_train)
train_file.close()

train_file = open('query.txt', 'w')
train_file.writelines(data_list_query)
train_file.close()


train_file = open('gallery.txt', 'w')
train_file.writelines(data_list_gallery)
train_file.close()

#print("Query_data_size:", len(data_list_query))
#print("Gallery_data_size:", len(data_list_gallery))
#query_file =