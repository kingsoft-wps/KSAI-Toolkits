export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
cd ../
mkdir -p output/aarch64
cd output/aarch64
cmake -D BUILD_TARGET_ARM=ON ../../
make 
cd ../../
