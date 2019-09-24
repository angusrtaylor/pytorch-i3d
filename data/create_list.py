import os    

def replace_path_in_file(file_in):
    with open(file_in, "rt") as fin:
        with open(file_in.replace('flow','rgb'), "wt") as fout:
            for line in fin:
                if "invalid" not in line:
                    fout.write(line.replace('/home/project/I3D/data/HMDB_FLOW', '/largedata/i3d/videos'))

if __name__ == '__main__':
    replace_path_in_file('train_flow.list')
    replace_path_in_file('test_flow.list')
