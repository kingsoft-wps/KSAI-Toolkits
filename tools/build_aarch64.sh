ARMCC_PREFIX=${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-
ARMCC_FLAGS="-funsafe-math-optimizations"
cd ../
mkdir -p output/aarch64
cd output/aarch64
cmake ../../ -B ../../output/aarch64  \
      -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
      -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
      -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DBUILD_TARGET_ARM=ON  \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64
make
cd ../../
