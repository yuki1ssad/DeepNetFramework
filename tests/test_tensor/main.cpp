#include<iostream>
#include<gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // google::ParseCommandLineFlags(&argc, &argv, true);
    // google::InitGoogleLogging(argv[0]);
    // google::SetStderrLogging(google::INFO);
    return RUN_ALL_TESTS();
}

// rm -rf * && cmake .. -DCMAKE_VERBOSE_MAKEFILE=on && make
// rm -rf * && cmake .. && make