import csv
import sys
CSV_FILE_NAME = '/home/holab/catkin_ws/src/sofa_gazebo_interface/vitaclink_gazebo/scripts/init_data.csv'

class CSV_Parse:
    def __init__(self):
        pass

    def get_node_id(self, csv_name, key = "index"):
        
        with open(csv_name, 'r') as csv_file:
            node_id = []
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                node_id.append(int(line[key]))
        
        return node_id

    def get_num_of_depth(self, csv_name, key = "number_depth"):

        with open(csv_name, 'r') as csv_file:
            num_of_depth = []
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                num_of_depth.append(int(line[key]))
        
        return num_of_depth

    def get_polar_coordinate(self, csv_name):
        pass
    def get_cartesian_coordinate(self, csv_name):
        pass

if __name__ == "__main__":
    csv_parse = CSV_Parse()
    node_id = csv_parse.get_node_id(CSV_FILE_NAME)
    num_of_depth = csv_parse.get_num_of_depth(CSV_FILE_NAME)
    print(node_id.index(7964))
