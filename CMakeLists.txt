cmake_minimum_required(VERSION 3.5)
project(slate_simulator)

set(CMAKE_CXX_STANDARD 14)
# Required to bring FindFlatBuffers in, need a better solution here. (TODO)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/reinforcement_learning/cmake/Modules/")

add_subdirectory(reinforcement_learning)

add_executable(simulator simulator.cc)
target_link_libraries(simulator PUBLIC rlclientlib)

add_executable(cbsimulator cb_simulator.cc)
target_link_libraries(cbsimulator PUBLIC rlclientlib)