cd ..
mkdir -p output/loongarch64
cd output/loongarch64
cmake -D BUILD_TARGET_LOONGARCH64=ON -D CMAKE_BUILD_TYPE=Release ../../
make
cd ../../
