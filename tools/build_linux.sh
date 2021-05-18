cd ..
mkdir -p output/linux
cd output/linux
cmake -D BUILD_TARGET_LINUX=ON ../../
make
cd ../../
