# cmake/SetRuntimeOutputDirectory.cmake

##########################
# From gpt.wustl.edu:

# Check if the GLOBAL_RUNTIME_OUTPUT_DIR is set
#function(set_runtime_output_directory target_dir)
#    if(GLOBAL_RUNTIME_OUTPUT_DIR)
#        # Set runtime output directory for the given directory
#        #    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOBAL_RUNTIME_OUTPUT_DIR}/${target_dir})
#        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${target_dir})
#        #    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)
#    endif()
#endfunction()
##########################
# From gpt.wustl.edu:
# function(set_target_output_directories subdir)
#     if(GLOBAL_RUNTIME_OUTPUT_DIR)
#         # Iterate over all targets defined so far
#         get_property(targets DIRECTORY ${subdir} PROPERTY BUILDSYSTEM_TARGETS)
#         foreach(target IN LISTS targets)
#             get_target_property(type ${target} TYPE)
#             # Apply only to executables (may include other types if necessary)
#             if (type STREQUAL "EXECUTABLE")
#                 set_target_properties(${target} PROPERTIES
#                     #                    RUNTIME_OUTPUT_DIRECTORY ${GLOBAL_RUNTIME_OUTPUT_DIR}/${subdir}
#                     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
#                 )
#             endif()
#         endforeach()
#     endif()
# endfunction()
##########################
# Function to set runtime output directory for a specific target
function(set_runtime_output_directory target)
    if(GLOBAL_RUNTIME_OUTPUT_DIR)
        # Use a directory structure under GLOBAL_RUNTIME_OUTPUT_DIR matching the source directory layout
        #        set(target_output_dir ${GLOBAL_RUNTIME_OUTPUT_DIR}/${CMAKE_CURRENT_SOURCE_DIR})
        #        message("CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}, target: ${target}")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(target_output_dir "${CMAKE_CURRENT_LIST_DIR}/bin/debug")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(target_output_dir "${CMAKE_CURRENT_LIST_DIR}/bin/release")
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            set(target_output_dir "${CMAKE_CURRENT_LIST_DIR}/bin/relwithdebinfo")
        elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
            set(target_output_dir "${CMAKE_CURRENT_LIST_DIR}/bin/minsizerel")
        else()
            message(WARNING "Unknown build type: ${CMAKE_BUILD_TYPE}")
            set(target_output_dir "${CMAKE_CURRENT_LIST_DIR}/bin/unknown")
        endif()

        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${target_output_dir}
        )
    endif()
endfunction()
###########################
