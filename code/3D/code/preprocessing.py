import os
import sys
import time
from multiprocessing import Pool

from processor_ligand import PDBbindDataProcessor

'''
Script to run data processor via multi-processing
In bash, run the following script:

python preprocessor.py {DATA_DIR} {SAVE_DIR} {NCPU}
'''


########1. Initialize data processor
# data_dir = sys.argv[1]
# save_dir = sys.argv[2]

#处理原来的数据
# data_dir = "/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/3D-molecular/DeepICL-master/data/PDBbind"
# save_dir = "/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/3D-molecular/DeepICL-master/data/PDBbind_PRO_SAVE_DIR"

data_dir = "/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/example"
save_dir = "/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/example_save"

print(f"save_dir: {save_dir}")
if len(os.listdir(save_dir)) > 0:
    token = input(f"Remove files in {save_dir}?: (y/n)")
    if token == "y":
        os.system(f"rm {save_dir}/*")
        time.sleep(5.0)
    elif token == "n":
        pass
    else:
        print("Wrong input:", token)
        exit()

# data_dir = 'data/PDBbind'
# save_dir = 'data/SAVE_DIR'




# 2. Run preprocessor
if __name__ == '__main__':

    preprocessor = PDBbindDataProcessor(
    data_dir=data_dir,
    save_dir=save_dir,
    max_atom_num=50, #50
    max_add_atom_num=30, #30
    use_whole_protein=False,
    predefined_scaffold=None,)

    # print("========debug=======")
    # exit()#调试代码停止语句

    print("NUM DATA:", preprocessor.num_data)
    time.sleep(2.0)
    
    st = time.time()
    pool = Pool(processes=8)  #int(sys.argv[3])是指-进程数{NCPU}
    r = pool.map_async(preprocessor.run, list(range(preprocessor.num_data)))
    r.wait()
    pool.close()
    pool.join()

    # 3. Print the result
    print("NUM PROCESSED DATA:", len(os.listdir(preprocessor.save_dir)))
    print("PROCESSING TIME:", f"{time.time() - st:.1f}", "(s)")
