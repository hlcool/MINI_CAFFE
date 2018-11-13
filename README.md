### ubt_caffe


### Build on Linux

Install OpenBLAS and protobuf library through system package manager. Or you can compile OpenBLAS and protobuf by yourself.

```
$ sudo apt install libopenblas-dev libprotobuf-dev protobuf-compiler
$ ./generatepb.sh
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j4
```

If you don't use Ubuntu, then you may need to install OpenBLAS and protobuf through your system package manager if any.
