'''
Author: myyao
Date: 2021-06-20 17:22:40
Description: 
'''
import os

if __name__ == '__main__':
    files = ["test_mobileNet.py", "test_nin.py", "test_resNet18.py", "test_vgg16.py"]
    # measure time in raspberry: 1.460, 0.209, 0.683, 2.283 
    for f in files:
        print("\n\n\n\n\nthe running file is {}".format(f))
        os.system("python {}".format(f))
    print("\n\n\n\n")
    print("-"*100)
    print("the script end ......")
    print("-"*100)