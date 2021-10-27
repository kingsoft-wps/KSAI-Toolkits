PROJECT(KSAI-Toolkits)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# target including Linux Windows Arm

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_PATH}/3rdparty
  ${PROJECT_SOURCE_PATH}/3rdparty/include
  ${PROJECT_SOURCE_PATH}/3rdparty/include/opencv3.4.6
  ${PROJECT_SOURCE_PATH}/include
  ${PROJECT_SOURCE_PATH}/src
)

SET(KSAI_TOOLKITS_SOURCE_FILES
  base/ks_common_op.cpp
  base/ks_inference_base.cpp
  base/ks_loader_op.cpp
)
if(WITH_OCR)
  list(APPEND KSAI_TOOLKITS_SOURCE_FILES
    OCR/ks_ocr_det.cpp
    OCR/ks_ocr_cls.cpp
    OCR/ks_ocr_recog.cpp
    OCR/ks_ocr_utility.cpp
    OCR/ks_ocr.cpp
    ${PROJECT_SOURCE_PATH}/3rdparty/clipper/clipper.cpp
  )
endif()

set(LINK_LIB_LIST "")
if(BUILD_TARGET_LINUX)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/opencv/libopencv_world.so
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/ksai-lite/libtensorflow-lite.so
  )
elseif(BUILD_TARGET_WINDOWS)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/windows_x86_64/opencv/opencv_world346.lib
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/windows_x86_64/ksai-lite/release/tensorflow-lite.lib
  )
elseif(BUILD_TARGET_ARM)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/aarch64/opencv/libopencv_world.so
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/aarch64/ksai-lite/libtensorflow-lite.so
    dl
)
elseif(BUILD_TARGET_MIPS)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/mips64/opencv/libopencv_world.so
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/mips64/ksai-lite/libtensorflow-lite.so
    dl
  )
else() # default is linux
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/opencv/libopencv_world.so
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/ksai-lite/libtensorflow-lite.so
  )
endif()

link_libraries(${LINK_LIB_LIST})
if(MSVC)
  source_group_by_dir(KSAI_TOOLKITS_SOURCE_FILES)
endif()
# shared library
ADD_LIBRARY(KSAI-Toolkits SHARED ${KSAI_TOOLKITS_SOURCE_FILES})

SET_TARGET_PROPERTIES(KSAI-Toolkits PROPERTIES OUTPUT_NAME "KSAI-Toolkits")

