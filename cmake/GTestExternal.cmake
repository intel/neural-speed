# https://stackoverflow.com/a/31120413
include(ExternalProject)

# variables to help keep track of gtest paths
set(GTEST_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/gtest")
set(GTEST_LOCATION "${GTEST_PREFIX}/src/GTestExternal-build")
set(GTEST_INCLUDES "${GTEST_PREFIX}/src/GTestExternal/googletest/include")
message(STATUS "GTEST_PREFIX=${GTEST_PREFIX}")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# external project download and build (no install for gtest)
ExternalProject_Add(GTestExternal
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW ON

    PREFIX "${GTEST_PREFIX}"

    # cmake arguments
    CMAKE_ARGS -Dgtest_force_shared_crt=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}

    BUILD_BYPRODUCTS ${GTEST_LOCATION}/lib/gtest.lib ${GTEST_LOCATION}/lib/gtest_main.lib

    # Disable install step
    INSTALL_COMMAND ""

    # Do not rebuild: https://stackoverflow.com/a/64994225
    UPDATE_COMMAND ""

    # Wrap download, configure and build steps in a script to log output
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    )

# variables defining the import location properties for the generated gtest and gtestmain libraries
set(GTEST_IMPORTED_LOCATION
    IMPORTED_LOCATION                 "${GTEST_LOCATION}/lib/gtest.lib")
set(GTESTMAIN_IMPORTED_LOCATION
    IMPORTED_LOCATION                 "${GTEST_LOCATION}/lib/gtest_main.lib")
message(STATUS "GTEST_IMPORTED_LOCATION=${GTEST_IMPORTED_LOCATION}")
message(STATUS "GTESTMAIN_IMPORTED_LOCATION=${GTESTMAIN_IMPORTED_LOCATION}")

# the gtest include directory exists only after it is build, but it is used/needed
# for the set_target_properties call below, so make it to avoid an error
file(MAKE_DIRECTORY ${GTEST_INCLUDES})

# define imported library GTest
add_library(GTest IMPORTED STATIC GLOBAL)
set_target_properties(GTest PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES     "${GTEST_INCLUDES}"
    IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
    ${GTEST_IMPORTED_LOCATION}
    )

# define imported library GTestMain
add_library(GTestMain IMPORTED STATIC GLOBAL)
set_target_properties(GTestMain PROPERTIES
    IMPORTED_LINK_INTERFACE_LIBRARIES GTest
    ${GTESTMAIN_IMPORTED_LOCATION}
    )

# make GTest depend on GTestExternal
add_dependencies(GTest GTestExternal)
