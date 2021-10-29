MIPS64CC_PREFIX=mips64el-linux-gnuabi64-
cd ../
mkdir -p output/mips64
cd output/mips64
cmake ../../ -B ../../output/mips64 \
	-DCMAKE_C_COMPILER=${MIPS64CC_PREFIX}gcc \
	-DCMAKE_CXX_COMPILER=${MIPS64CC_PREFIX}g++ \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DCMAKE_SYSTEM_NAME=Linux \
	-DCMAKE_SYSTEM_PROCESSOR=mips64el\
	-DBUILD_TARGET_MIPS=ON
make
cd ../../
