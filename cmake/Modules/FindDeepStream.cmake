# FindDeepStream.cmake - Version agnostic

include(FindPackageHandleStandardArgs)

# Allow user to override DeepStream root dir before searching
if(NOT DEEPSTREAM_ROOT_DIR)
    # Base search paths
    set(DEEPSTREAM_BASE_PATHS
        /opt/nvidia/deepstream
        /usr/local/deepstream
    )

    # Find the newest DeepStream installation by looking for nvds_version.h
    find_path(DEEPSTREAM_ROOT_DIR
        NAMES sources/includes/nvds_version.h
        PATHS ${DEEPSTREAM_BASE_PATHS}
        PATH_SUFFIXES deepstream-6.3 deepstream-6.2 deepstream-6.1 deepstream-6.0 deepstream
        DOC "Root directory of DeepStream installation"
    )
endif()

# Fail early if no root dir found
if(NOT DEEPSTREAM_ROOT_DIR)
    set(DEEPSTREAM_ROOT_DIR "")
    set(DeepStream_FOUND FALSE)
    find_package_handle_standard_args(DeepStream
        REQUIRED_VARS DEEPSTREAM_ROOT_DIR
        VERSION_VAR DEEPSTREAM_VERSION
    )
    return()
endif()

# Extract version from installation path folder name
get_filename_component(DEEPSTREAM_DIR_NAME ${DEEPSTREAM_ROOT_DIR} NAME)
string(REGEX MATCH "deepstream-([0-9]+\\.[0-9]+)" _match ${DEEPSTREAM_DIR_NAME})

if(_match)
    string(REGEX REPLACE "deepstream-([0-9]+\\.[0-9]+)" "\\1" DEEPSTREAM_VERSION ${DEEPSTREAM_DIR_NAME})
else()
    # fallback: try to parse version from nvds_version.h
    set(NVDS_VERSION_HEADER "${DEEPSTREAM_ROOT_DIR}/sources/includes/nvds_version.h")
    if(EXISTS ${NVDS_VERSION_HEADER})
        file(READ ${NVDS_VERSION_HEADER} NVDS_VERSION_CONTENTS)
        string(REGEX MATCH "#define[ \t]+NVDS_VERSION_MAJOR[ \t]+([0-9]+)" _major_match "${NVDS_VERSION_CONTENTS}")
        string(REGEX MATCH "#define[ \t]+NVDS_VERSION_MINOR[ \t]+([0-9]+)" _minor_match "${NVDS_VERSION_CONTENTS}")
        if(_major_match AND _minor_match)
            string(REGEX REPLACE "#define[ \t]+NVDS_VERSION_MAJOR[ \t]+([0-9]+)" "\\1" NVDS_VERSION_MAJOR "${_major_match}")
            string(REGEX REPLACE "#define[ \t]+NVDS_VERSION_MINOR[ \t]+([0-9]+)" "\\1" NVDS_VERSION_MINOR "${_minor_match}")
            set(DEEPSTREAM_VERSION "${NVDS_VERSION_MAJOR}.${NVDS_VERSION_MINOR}")
        else()
            set(DEEPSTREAM_VERSION "unknown")
        endif()
    else()
        set(DEEPSTREAM_VERSION "unknown")
    endif()
endif()

# Find include directory
find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvds_version.h
    PATHS ${DEEPSTREAM_ROOT_DIR}/sources/includes
    DOC "DeepStream include directory"
)

# List of libraries to find
set(DEEPSTREAM_LIB_NAMES
    nvds_meta
    nvds_infer
    nvdsgst_meta
    nvdsgst_helper
    nvds_yml_parser
    nvbufsurface
    nvbufsurftransform
)

# Search for libraries (try unversioned and versioned)
foreach(LIB ${DEEPSTREAM_LIB_NAMES})
    find_library(DEEPSTREAM_${LIB}_LIB
        NAMES ${LIB} ${LIB}.${DEEPSTREAM_VERSION}
        PATHS
            ${DEEPSTREAM_ROOT_DIR}/lib
            ${DEEPSTREAM_ROOT_DIR}/lib/x86_64-linux-gnu
        NO_DEFAULT_PATH
        DOC "DeepStream library: ${LIB}"
    )
endforeach()

# Mark required vars for find_package_handle_standard_args
set(REQUIRED_VARS DEEPSTREAM_INCLUDE_DIR DEEPSTREAM_nvds_meta_LIB DEEPSTREAM_nvds_infer_LIB)

find_package_handle_standard_args(DeepStream
    REQUIRED_VARS ${REQUIRED_VARS}
    VERSION_VAR DEEPSTREAM_VERSION
)

if(DeepStream_FOUND)
    set(DeepStream_INCLUDE_DIRS ${DEEPSTREAM_INCLUDE_DIR} CACHE PATH "DeepStream include dirs")
    set(DeepStream_LIBRARIES "" CACHE STRING "DeepStream libraries")

    # Create imported targets for each found library
    foreach(LIB ${DEEPSTREAM_LIB_NAMES})
        if(DEEPSTREAM_${LIB}_LIB)
            add_library(DeepStream::${LIB} SHARED IMPORTED)
            set_target_properties(DeepStream::${LIB} PROPERTIES
                IMPORTED_LOCATION "${DEEPSTREAM_${LIB}_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${DeepStream_INCLUDE_DIRS}"
            )
            list(APPEND DeepStream_LIBRARIES DeepStream::${LIB})
        else()
            message(WARNING "DeepStream library not found: ${LIB}")
        endif()
    endforeach()

    # Expose variables for usage
    set(DeepStream_INCLUDE_DIRS ${DeepStream_INCLUDE_DIRS} CACHE PATH "DeepStream include dirs" FORCE)
    set(DeepStream_LIBRARIES ${DeepStream_LIBRARIES} CACHE STRING "DeepStream libraries" FORCE)

    message(STATUS "Found DeepStream ${DEEPSTREAM_VERSION} at: ${DEEPSTREAM_ROOT_DIR}")
    message(STATUS "  Includes: ${DeepStream_INCLUDE_DIRS}")
    message(STATUS "  Libraries found:")
    foreach(LIB ${DEEPSTREAM_LIB_NAMES})
        if(DEEPSTREAM_${LIB}_LIB)
            message(STATUS "    - ${LIB}: ${DEEPSTREAM_${LIB}_LIB}")
        else()
            message(WARNING "    - ${LIB}: NOT FOUND")
        endif()
    endforeach()
endif()

mark_as_advanced(
    DEEPSTREAM_INCLUDE_DIR
    DEEPSTREAM_ROOT_DIR
    DEEPSTREAM_nvds_meta_LIB
    DEEPSTREAM_nvds_infer_LIB
)


