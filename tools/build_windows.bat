cd ../
mkdir output
cd output/
mkdir windows
cd windows/
cmake -DBUILD_TARGET_WINDOWS=ON ../../
cmake --build . --config Release
copy lib\Release\KSAI-ToolKits.dll bin\Release\KSAI-ToolKits.dll
cd ../../
