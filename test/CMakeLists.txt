PROJECT(KSAI-Toolkits-demo)

SET(KSAI_TOOLKITS_DEMO_SOURCE_FILES
)
if(WITH_OCR)
  list(APPEND KSAI_TOOLKITS_DEMO_SOURCE_FILES
    test_ocr.cpp
  )
endif()

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_PATH}/3rdparty/include/opencv3.4.6
  ${PROJECT_SOURCE_PATH}/include
)

set(LINK_LIB_LIST "")
if(BUILD_TARGET_LINUX)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/opencv/libopencv_world.so
    ${LIBRARY_OUTPUT_PATH}/libKSAI-Toolkits.so
  )
elseif(BUILD_TARGET_WINDOWS)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/windows_x86_64/opencv/opencv_world346.lib
    ${LIBRARY_OUTPUT_PATH}/Release/KSAI-Toolkits.lib
  )
elseif(BUILD_TARGET_ARM)
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/aarch64/others/libz.so.1
    ${LIBRARY_OUTPUT_PATH}/libKSAI-Toolkits.so
  )
else() # default is linux
  set(LINK_LIB_LIST
    ${PROJECT_SOURCE_PATH}/3rdparty/libs/linux_x86_64/opencv/libopencv_world.so
    ${LIBRARY_OUTPUT_PATH}/libKSAI-Toolkits.so
  )
endif()
link_libraries(${LINK_LIB_LIST})

ADD_EXECUTABLE(KSAI_Toolkits_demo ${KSAI_TOOLKITS_DEMO_SOURCE_FILES})

if(BUILD_TARGET_WINDOWS)
  file(COPY  "${PROJECT_SOURCE_PATH}/3rdparty/libs/windows_x86_64/opencv/opencv_world346.dll" DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Release/")
  file(COPY  "${PROJECT_SOURCE_PATH}/3rdparty/libs/windows_x86_64/ksai-lite/release/tensorflow-lite.dll" DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Release/")
endif()
