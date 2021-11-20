import os
import shutil
from parse_csv import CSV_Parse, CSV_FILE_NAME

csv_parse = CSV_Parse()
node_id = csv_parse.get_node_id(CSV_FILE_NAME)
num_of_depth = csv_parse.get_num_of_depth(CSV_FILE_NAME)

# root_dir = '/media/holab/SSD-PGU3/Nhan/sofadata'
root_dir = '/media/holab/SSD-PGU3/skin_state_data'

img_count = 0
for idx, num in zip(node_id[:], num_of_depth[:]):
    for i in range(1,num+1):
        img_count += 1
        state_id = str(img_count).zfill(5)
        skin_fname_old = "skin{0:05}.stl".format(img_count)
        marker_fname_old = "marker{0:05}.stl".format(img_count)
        os.rename(os.path.join(root_dir,skin_fname_old), os.path.join(src_skin_dir,'skin{0}_{1:02}'.format(idx,i)))
        os.rename(os.path.join(root_dir,marker_fname_old), os.path.join(src_marker_dir,'marker{0}_{1:02}'.format(idx,i)))
        # shutil.copy(os.path.join(src_skin_dir,skin_file_name), target_dir)
        # shutil.copy(os.path.join(src_marker_dir,marker_file_name), target_dir)

print('The total number of virtual tranning images: {0}'.format(img_total))


# root_dir = '/media/ho-lab/SSD-PGU3/skin_state_data'

# for file in os.listdir(root_dir):
#     ls = file.split('_')
#     fname = ls[0] + '_' + ls[1].zfill(2) + '.stl'
#     os.rename(os.path.join(root_dir, file), os.path.join(root_dir, fname))

