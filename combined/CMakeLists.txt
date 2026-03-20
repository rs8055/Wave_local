cmake_minimum_required(VERSION 3.13.4)

set(TARGET "combined")

# Force static Boost to avoid runtime conflicts with other projects
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_SYSTEM_PATHS OFF)

# Find deal.II - standard apt install locations
find_package(deal.II 9.7.1
  HINTS
    /usr/lib/x86_64-linux-gnu/cmake/deal.II
    /usr/share/deal.II
    /usr/lib/cmake/deal.II
    ${deal.II_DIR}
    ${DEAL_II_DIR}
    $ENV{DEAL_II_DIR}
  REQUIRED
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "Try: cmake .. -DDEAL_II_DIR=/path/to/deal.II\n"
  )
endif()

if(NOT DEAL_II_WITH_LAPACK)
  message(FATAL_ERROR "This project requires deal.II with LAPACK support.")
endif()

deal_ii_initialize_cached_variables()

project(${TARGET})

set(TARGET_SRC main.cpp)

include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/source
)

deal_ii_invoke_autopilot()

# Link Boost statically to avoid conflicts with HArDCore3D's Boost
find_package(Boost REQUIRED COMPONENTS program_options)
target_link_libraries(${TARGET} Boost::program_options)
