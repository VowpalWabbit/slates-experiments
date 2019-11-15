cmake_minimum_required(VERSION 3.5)
project(slate_simulator)

set(CMAKE_CXX_STANDARD 17)
# Required to bring FindFlatBuffers in, need a better solution here. (TODO)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/reinforcement_learning/cmake/Modules/")

add_subdirectory(reinforcement_learning)

add_executable(cb_skype_slate_simulator cb_skype_slate_simulator.cc)
target_link_libraries(cb_skype_slate_simulator PUBLIC rlclientlib)

add_executable(skype_slate_simulator skype_slate_simulator.cc)
target_link_libraries(skype_slate_simulator PUBLIC rlclientlib)

add_executable(cb_simulator cb_simulator.cc)
target_link_libraries(cb_simulator PUBLIC rlclientlib)

add_executable(cb_slate_simulator cb_slate_simulator.cc)
target_link_libraries(cb_slate_simulator PUBLIC rlclientlib)
