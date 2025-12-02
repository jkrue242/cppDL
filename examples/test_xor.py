# in order to import this, you need to first build the entire project 
# this will create a cpDLL.so shared library file that python can load
# and thus lets you import it here.

# todo: see if i can use this to run CUDA on google colab?
import cpDLL 
cpDLL.test_xor()