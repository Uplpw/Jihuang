# headers
set(PYTHON_INCLUDE_DIRS $ENV{JIHUANG_PYTHON_INCLUDE_DIRS})
set(Boost_INCLUDE_DIRS $ENV{JIHUANG_BOOST_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/code ${PROJECT_SOURCE_DIR})



# libraries
set(PYTHON_LIBRARIES $ENV{JIHUANG_PYTHON_LD_LIB_DIRS})
set(Boost_LIBRARIES $ENV{JIHUANG_BOOST_LD_LIB_DIRS})
#set(GAME_LIBRARY ${PROJECT_SOURCE_DIR}/jihuang/libzhuozhu.so)

# cpp codes
#set(py_zhuozhu_src ${PROJECT_SOURCE_DIR}/python/zhuozhu/_zhuozhu.cpp)
set(py_jihuang_src ${PROJECT_SOURCE_DIR}/python/jihuang/_jihuang.cpp)

# compile
#add_library(pyzhuozhu SHARED ${py_zhuozhu_src})
#set_target_properties(pyzhuozhu PROPERTIES PREFIX "" OUTPUT_NAME "_zhuozhu")
#target_link_libraries(pyzhuozhu zhuozhu ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

add_library(pyjihuang SHARED ${py_jihuang_src})
set_target_properties(pyjihuang PROPERTIES PREFIX "" OUTPUT_NAME "_jihuang")
target_link_libraries(pyjihuang jihuang ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

# ln -sf _game.so
#set(__zhuozhu_linkname "${PROJECT_SOURCE_DIR}/python/zhuozhu/_zhuozhu.so")
#add_custom_command(TARGET pyzhuozhu POST_BUILD
#                   COMMAND ln -sf $<TARGET_LINKER_FILE:pyzhuozhu> "${__zhuozhu_linkname}"
#                   COMMENT "Creating symlink _zhuozhu.so")

set(__jihuang_linkname "${PROJECT_SOURCE_DIR}/python/jihuang/_jihuang.so")
add_custom_command(TARGET pyjihuang POST_BUILD
                   COMMAND ln -sf $<TARGET_LINKER_FILE:pyjihuang> "${__jihuang_linkname}"
                   COMMENT "Creating symlink _jihuang.so")

set(__jihuang_gymapi_link "${PROJECT_SOURCE_DIR}/python/gym_api/Jihuang/_jihuang.so")
add_custom_command(TARGET pyjihuang POST_BUILD
                    COMMAND ln -sf $<TARGET_LINKER_FILE:pyjihuang> "${__jihuang_gymapi_link}"
                    COMMENT "Creating symlink _jihuang.so")
