# 3.10 for CUDA language support
# 3.17 for FindCUDAToolkit
# 3.18 for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE RelWithDebInfo) # optimized plus debugging
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(teachingkit-labs CXX CUDA)

####################################
# from ChatGPT (gpt.wustl.edu): set executable target dir
# Set the global runtime output directory
set(GLOBAL_RUNTIME_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)

# Add the global include to set runtime output directory for all subdirectories
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(SetRuntimeOutputDirectory)
####################################

set(BUILD_LIBGPUTK_LIBRARY ON CACHE BOOL "Turn on build of libgputk library")
set(BUILD_SOLUTION ON CACHE BOOL "Turn on build of code solution")
set(BUILD_DESCRIPTION OFF CACHE BOOL "Turn on build of lab description")
set(BUILD_DATASET_GENERATOR ON CACHE BOOL "Turn on building of dataset generator")
set(BUILD_TEMPLATE OFF CACHE BOOL "Turn on build of code templates")
set(BUILD_JSONLOG OFF CACHE BOOL "Turn on build of detailed json logging")
set(BUILD_LOGTIME OFF CACHE BOOL "Turn on build of time logging")

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

if (BUILD_SOLUTION OR BUILD_TEMPLATE OR BUILD_LIBGPUTK_LIBRARY)
  find_package(CUDAToolkit REQUIRED)
else()
  find_package(CUDAToolkit)
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)

if (NOT WIN32)
    find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)

# allow C99 for cmake versions less than 3.1
set (CMAKE_C_STANDARD 99)

# set(CUDA_PROPAGATE_HOST_FLAGS TRUE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# https://cmake.org/cmake/help/v3.10/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html
# enable device static linking on all targets?
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# coles: below is for Ada-generation RTX 4000 GPUs on linuxlab cluster
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89")
set(CMAKE_CUDA_ARCHITECTURES 89)
message(STATUS ${CMAKE_CUDA_FLAGS})
if(NOT CMAKE_BUILD_TYPE)
     set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
# Set warning levels for host compilation
if (UNIX)
     add_compile_options(-Wno-unused-function)
     add_compile_options(-Wno-unused-parameter)
     add_compile_options(-Wextra)
     add_compile_options(-Wno-unused-variable)
endif ()

add_definitions(-DGPUTK_USE_CUDA)

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif(NOT GCC_EXECUTABLE)
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++-3.6 clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CMAKE_CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
    endif (${CMAKE_SYSTEM_VERSION} LESS 12.0)
endif (APPLE)

# Set the correct linker options on UNIX platforms
if (UNIX)
    set(LINK_LIBRARIES "-lm -lcuda")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else (APPLE)
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else (HAVE_CLOCK_GETTIME)
            message(FATAL_ERROR "clock_gettime() not found")
        endif (HAVE_CLOCK_GETTIME)
    endif (APPLE)
endif (UNIX)

macro(add_lab_helper NAME SOURCES)
    add_executable(${NAME} ${SOURCES})
    target_link_libraries(${NAME} ${GPUTKLIB} ${LINK_LIBRARIES})
    target_link_libraries(${NAME} CUDA::cudart)
endmacro(add_lab_helper)

macro(add_lab_solution NAME SOURCES)
    if (BUILD_SOLUTION)
        add_lab_helper("${NAME}_Solution" ${SOURCES})
        set_runtime_output_directory("${NAME}_Solution")
    endif(BUILD_SOLUTION)
endmacro(add_lab_solution)

macro(add_lab_template NAME SOURCES)
    if (BUILD_TEMPLATE)
        add_lab_helper("${NAME}_Template" ${SOURCES})
        set_runtime_output_directory("${NAME}_Template")
    endif(BUILD_TEMPLATE)
endmacro(add_lab_template)

macro(add_lab_libs NAME LIBRARIES)
    target_link_libraries(${NAME} ${LIBRARIES})
endmacro(add_lab_libs)

macro(add_lab_solution_libs NAME LIBRARIES)
    if (BUILD_SOLUTION)
        add_lab_libs("${NAME}_Solution" ${LIBRARIES})
    endif(BUILD_SOLUTION)
endmacro(add_lab_solution_libs)

macro(add_lab_template_libs NAME LIBRARIES)
    if (BUILD_TEMPLATE)
        add_lab_libs("${NAME}_Template" ${LIBRARIES})
    endif(BUILD_TEMPLATE)
endmacro(add_lab_template_libs)

macro(add_generator NAME SOURCES)
  if (BUILD_DATASET_GENERATOR)
    add_executable("${NAME}_DatasetGenerator" ${SOURCES})
    target_link_libraries("${NAME}_DatasetGenerator" ${GPUTKLIB} ${LINK_LIBRARIES})
    target_link_libraries("${NAME}_DatasetGenerator" CUDA::cudart)
    set_runtime_output_directory("${NAME}_DatasetGenerator")
  endif (BUILD_DATASET_GENERATOR)
endmacro(add_generator)

macro(add_lab NAME)
    if (BUILD_DESCRIPTION)
        find_program(PANDOC pandoc)
        set(MAKE_LAB_HANDOUT_PY "${PROJECT_SOURCE_DIR}/support/make_lab_handout.py")
        add_custom_target(
            "Build${NAME}Description"
            COMMAND python ${MAKE_LAB_HANDOUT_PY}
                            ${CMAKE_CURRENT_LIST_DIR}
                            ${PROJECT_SOURCE_DIR}/support
                            ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_custom_target("${NAME}Description" ALL echo DEPENDS "Build${NAME}Description")
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.pdf)
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${NAME}.docx)
    endif (BUILD_DESCRIPTION)
endmacro(add_lab)

include_directories(${PROJECT_SOURCE_DIR}/libgputk)

if (BUILD_LIBGPUTK_LIBRARY)
    include(libgputk/sources.cmake)
    add_library(${GPUTKLIB} ${LIBGPUTK_SOURCE_FILES}) # TODO: static?
    foreach(_src ${LIBGPUTK_SOURCE_FILES})
        set_source_files_properties(${_src} PROPERTIES LANGUAGE CUDA)
    endforeach()
    if (WIN32)
        target_link_libraries(${GPUTKLIB} ${CUDA_cuda_LIBRARY})
    endif (WIN32)
    target_link_libraries(${GPUTKLIB} ${LINK_LIBRARIES})
    target_link_libraries(${GPUTKLIB} CUDA::cudart)
    if (BUILD_LOGTIME)
        set_target_properties(${GPUTKLIB} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -DgpuTKLogger_TimerLog" )
    endif()
    if (BUILD_JSONLOG)
        set_target_properties(${GPUTKLIB} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -DgpuTKLogger_printOnLog" )
    endif()
endif (BUILD_LIBGPUTK_LIBRARY)

# ======================================================
# Single‑lab build: Convolution (flat source structure)
# ======================================================
add_lab_solution("Convolution" solution.cu)
add_generator("Convolution" dataset_generator.cpp)
set_source_files_properties(dataset_generator.cpp PROPERTIES LANGUAGE CUDA)

# Uncomment the following if you copied the description files (markdown, pdf, etc.)
# add_lab("Convolution")
