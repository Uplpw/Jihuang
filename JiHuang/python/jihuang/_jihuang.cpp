#include <iostream>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vector>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <string>
#include "jihuang/external/env.h"

void InitLog(const std::string log_dir, int log_level, bool log_tostderr) {
#ifdef USE_GLOG
    std::cout << "Init Log Start. " << FLAGS_log_dir << std::endl;
    google::InitGoogleLogging("zhuozhu");
    google::InstallFailureSignalHandler();

    FLAGS_log_dir = log_dir;
    FLAGS_minloglevel = log_level;
    FLAGS_alsologtostderr = log_tostderr;
    FLAGS_colorlogtostderr = true;

    VLOG(0) << "VLOG(0)";
    VLOG(1) << "VLOG(1)";
    VLOG(2) << "VLOG(2)";
    for (int i = 0; i < 100; ++i) {
        VLOG(300) << "Init Log Done!";
    }
    std::cout << "Init Log Finish. " << FLAGS_log_dir << std::endl;
#endif
}

void Log(const std::string& s) {
#ifdef USE_GLOG
    VLOG(300) << s;
#endif
}

template <class T>
std::vector<T> py_to_vector(boost::python::list pyiter) {
    std::vector<T> vec;
    for (int i = 0; i < len(pyiter); ++i) {
        vec.push_back(boost::python::extract<T>(pyiter[i]));
    }
    return vec;
}

template <class T>
std::vector<std::vector<T> > py_to_vector_2d(boost::python::list pylist) {
    std::vector<std::vector<T> > vec;
    for (int i = 0; i < len(pylist); ++i) {
        boost::python::list l1 = boost::python::extract<boost::python::list>(pylist[i]);
        std::vector<T> vec1;
        for (int j = 0; j < len(l1); ++j) {
            vec1.push_back(boost::python::extract<T>(l1[j]));
        }
        vec.push_back(vec1);
    }
    return vec;
}

template <class T>
boost::python::list vector_to_pylist(std::vector<T> vec) {
    boost::python::list ob;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        ob.attr("append")(vec[i]);
    }
    return ob;
}

template <class T>
boost::python::list vector_to_pylist_2d(std::vector<std::vector<T> > vec) {
    boost::python::list l1;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        boost::python::list l2;
        for (std::size_t j = 0; j < vec[i].size(); ++j) {
            l2.attr("append")(vec[i][j]);
        }
        l1.attr("append")(l2);
    }
    return l1;
}

/*
boost::python::object EnvGetMap(Environment* env_ptr) {
    std::vector<std::vector<float> > mp = env_ptr->GetMap();
    return vector_to_pylist_2d<float>(mp);
}
*/

void EnvStep(Environment* env_ptr, boost::python::list py_ob) {
    std::vector<std::vector<float> > action = py_to_vector_2d<float>(py_ob);
    env_ptr->Step(action);
}

void EnvReset(Environment* env_ptr) {
    env_ptr->Reset();
}

void EnvSetCurrentGameID(Environment* env_ptr, const int targetID) {
    env_ptr->SetCurrentGameID(targetID);
}

boost::python::object EnvGetAgentObserve(Environment* env_ptr) {
    std::vector<std::vector<float> > ob = env_ptr->GetAgentObserve();
    return vector_to_pylist_2d<float>(ob);
}

/*
boost::python::object EnvGetAgentReward(Environment* env_ptr) {
    std::vector<float> rw = env_ptr->GetAgentReward();
    return vector_to_pylist<float>(rw);
}
*/

boost::python::object EnvGetInitializeMap(Environment* env_ptr) {
    std::vector<float> im = env_ptr->GetInitializeMap();
    return vector_to_pylist<float>(im);
}

/*
boost::python::object EnvGetGetAllObjectInMap(Environment* env_ptr) {
    std::vector<float> im = env_ptr->GetAllObjectInMap();
    return vector_to_pylist<float>(im);
}
*/

boost::python::object EnvGetShowEnvironmentInformation(Environment* env_ptr) {
    std::vector<float> sei = env_ptr->GetShowEnvironmentInformation();
    return vector_to_pylist<float>(sei);
}

boost::python::object EnvGetMapInformation(Environment* env_ptr) {
    std::vector<float> info = env_ptr->GetMapInformation();
    return vector_to_pylist<float>(info);
}


boost::python::object EnvGetMoveInformation(Environment* env_ptr) {
    std::vector<int> smi = env_ptr->GetMoveInformation();
    return vector_to_pylist<int>(smi);
}


unsigned int EnvGetWidth(Environment* env_ptr) {
    return env_ptr->GetWidth();
}


unsigned int EnvGetHeight(Environment* env_ptr) {
    return env_ptr->GetHeight();
}

unsigned int EnvGetAgentsNumber(Environment* env_ptr) {
    return env_ptr->GetAgentsNumber();
}

BOOST_PYTHON_MODULE(_jihuang) {
    boost::python::def("init_log", &InitLog);
    boost::python::def("log", &Log);
    boost::python::class_<Environment>("Env", boost::python::init<int, int, bool>())
        .def(boost::python::init<std::string, std::string, std::string, std::string, int>())
        // .def("get_map", &EnvGetMap) // get total info of the map, return a 2-D list;
            // [[is_mountain(0,0), is_mountain(0,1), ... is_mountain(0,H-1)],
            //  [is_mountain(1,0), is_mountain(1,1), ... is_mountain(1,H-1)],
            //  ... ,
            //  [is_mountain(W-1,0), is_mountain(W-1,1), ... is_mountain(W-1,H-1)],
            //  [type_of_obj_0(1 for human, 2 for pig), id_of_obj_0, x_of_obj_0, y_of_obj_0, obj_0_see_enemy, 
            //   type_of_obj_1, id_of_obj_1, x_of_obj_1, y_of_obj_1, obj_1_see_enemy, 
            //   ... ,
            //   type_of_obj_N-1, id_of_obj_N-1, x_of_obj_N-1, y_of_obj_N-1, obj_N-1_see_enemy]]
        .def("reset", &EnvReset)
        .def("set_current_game_id", &EnvSetCurrentGameID)
        .def("step", &EnvStep) // no return, input 2-D list of all humans decision.
        .def("get_agent_observe", &EnvGetAgentObserve) // get observation of each human, return a 2-D list 
            // of shape (agent_num, observe_info_dimension), length of observe_info_dimension differs for 
            // each human, and depends on the number of objects in vision range, each object has info of 3 
            // numbers; For example, format of observation_info: [idx_of_mine, type_of_1st_object_in_vision, 
            // x_coordinate_of_1st_object_in_vision,  y_coordinate_of_1st_object_in_vision, 
            // type_of_2nd_object_in_vision, ... ]
        // .def("get_agent_reward", &EnvGetAgentReward) // get reward from environment of each human
        .def("get_initialize_map", &EnvGetInitializeMap) // get initialize map information
        // .def("get_all_object_in_map", &EnvGetGetAllObjectInMap) // get initialize map information
        .def("get_show_environment_information", &EnvGetShowEnvironmentInformation) // get show information
        .def("get_move_information", &EnvGetMoveInformation) // get show information
        .def("get_map_information", &EnvGetMapInformation)
        .def("get_width", &EnvGetWidth) // width of map
        .def("get_height", &EnvGetHeight) // height of map
        .def("get_agents_number", &EnvGetAgentsNumber) // height of map
        ;
}
