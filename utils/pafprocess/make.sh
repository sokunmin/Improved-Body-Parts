rm -rf build/
rm pafprocess.py
rm pafprocess_wrap.cpp
rm pafprocess_wrap.cxx
rm _pafprocess.cpython-37m-x86_64-linux-gnu.so
swig -python -c++ pafprocess.i
python3 setup.py build_ext --inplace