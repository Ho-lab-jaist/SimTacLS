import os
from parse_csv import CSV_Parse, CSV_FILE_NAME

imbot_file = "/media/ho-lab/SSD-PGU3/iotouch/data/simulated_images/image_bot"
imup_file = "/media/ho-lab/SSD-PGU3/iotouch/data/simulated_images/image_up"

csv_parse = CSV_Parse()
node_id = csv_parse.get_node_id(CSV_FILE_NAME)
num_of_depth = csv_parse.get_num_of_depth(CSV_FILE_NAME)

for idx, num in zip(node_id, num_of_depth):
    name_id = str(idx)
    bot_path = os.path.join(imbot_file, name_id)
    up_path = os.path.join(imup_file, name_id)
    os.makedirs(bot_path)
    os.makedirs(up_path)
    for d in range(1, num+1):
        name_depth = str(d)
        os.makedirs(os.path.join(bot_path, name_depth))
        os.makedirs(os.path.join(up_path, name_depth))

print(os.listdir(imbot_file))
print(os.listdir(imup_file))