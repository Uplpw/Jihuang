import os

OPTIONS_LIST = [
    '-fauto-inc-dec',
    '-fbranch-count-reg',
    '-fcombine-stack-adjustments',
    '-fcompare-elim',
    '-fcprop-registers',
    '-fdce',
    '-fdefer-pop',
    '-fdelayed-branch',
    '-fdse',
    '-fforward-propagate',
    '-fguess-branch-probability',
    '-fif-conversion',
    '-fif-conversion2',
    '-finline-functions-called-once',
    '-fipa-modref',
    '-fipa-profile',
    '-fipa-pure-const',
    '-fipa-reference',
    '-fipa-reference-addressable',
    '-fmerge-constants',
    '-fmove-loop-invariants',
    '-fomit-frame-pointer',
    '-freorder-blocks',
    '-fshrink-wrap',
    '-fshrink-wrap-separate',
    '-fsplit-wide-types',
    '-fssa-backprop',
    '-fssa-phiopt',
    '-ftree-bit-ccp',
    '-ftree-ccp',
    '-ftree-ch',
    '-ftree-coalesce-vars',
    '-ftree-copy-prop',
    '-ftree-dce',
    '-ftree-dominator-opts',
    '-ftree-dse',
    '-ftree-forwprop',
    '-ftree-fre',
    '-ftree-phiprop',
    '-ftree-pta',
    '-ftree-scev-cprop',
    '-ftree-sink',
    '-ftree-slsr',
    '-ftree-sra',
    '-ftree-ter',
    '-funit-at-a-time',
]

if __name__=='__main__':
    cmd = "rm CMakeCache.txt && cmake .. && make -j && ./jihuang_main | grep consume"
    for option in OPTIONS_LIST:
        s = '''CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
PROJECT(virtual_earth)

SET(CMAKE_C_COMPILER $ENV{JIHUANG_CC})
SET(CMAKE_CXX_COMPILER $ENV{JIHUANG_CXX})

add_definitions(-D jihuang_VERSION=0.1)

# add_definitions(-D AUTO_AGENT)
# add_definitions(-D EXPLAIN_INFO)

SET(CMAKE_BUILD_TYPE Debug)

# gdb enable
IF (UNIX)
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS} %s -Wall -pg -ggdb --std=c++11 -pthread -fPIC")
    SET(CMAKE_CXX_FLAGS_RELEASE  "$ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS} -O3 -Wall --std=c++11 -pthread -fPIC")
ENDIF()

#FILE(GLOB_RECURSE code_sourcefiles "code/*.cpp" "jihuang/attr/*.cpp")
FILE(GLOB_RECURSE code_sourcefiles "src/code/*.cpp")

FILE(GLOB_RECURSE jihuang_sourcefiles "src/jihuang/*.cpp")
SET(jihuang_main_cc "src/jihuang/main.cc")
SET(jihuang_proto_cc "src/jihuang/jihuang.pb.cc")
SET(jihuang_proto_h "include/jihuang/jihuang.pb.h")

INCLUDE_DIRECTORIES(include)

# Gflags
INCLUDE_DIRECTORIES($ENV{JIHUANG_INCLUDE_GFLAG})
LINK_LIBRARIES($ENV{JIHUANG_LD_LIB_GFLAG})
# FIND_PACKAGE(gflags REQUIRED)
# LINK_LIBRARIES(gflags)


# Glog
INCLUDE_DIRECTORIES($ENV{JIHUANG_INCLUDE_GLOG})
LINK_LIBRARIES($ENV{JIHUANG_LD_LIB_GLOG})
# FIND_PACKAGE(glog REQUIRED)
# LINK_LIBRARIES(glog)

# ProtoBuf
INCLUDE_DIRECTORIES($ENV{JIHUANG_INCLUDE_PROTOBUF})
LINK_LIBRARIES($ENV{JIHUANG_LD_LIB_PROTOBUF})
#FIND_PACKAGE(Protobuf REQUIRED)
#LINK_LIBRARIES(Protobuf)

# BASE lib
#ADD_LIBRARY(base STATIC ${code_sourcefiles})
INCLUDE_DIRECTORIES(${headerfiles})

# jihuang lib
ADD_LIBRARY(jihuang STATIC ${jihuang_sourcefiles} ${jihuang_proto_cc} ${jihuang_proto_h} ${code_sourcefiles})
#TARGET_LINK_LIBRARIES(jihuang base)

SET_TARGET_PROPERTIES(jihuang PROPERTIES OUTPUT_NAME "jihuang")

# exec
ADD_EXECUTABLE(jihuang_main ${jihuang_main_cc})
TARGET_LINK_LIBRARIES(jihuang_main PRIVATE jihuang)

ADD_SUBDIRECTORY(python)
        ''' % option
        with open('../CMakeLists.txt', 'w') as fp:
            fp.write(s)
        os.system(cmd+f'>{option}.txt')