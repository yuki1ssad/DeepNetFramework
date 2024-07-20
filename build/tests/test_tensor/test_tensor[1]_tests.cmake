add_test([=[tensor.mdata]=]  /home/wangxuefei/workspace/DeepNetFramework/build/tests/test_tensor/test_tensor [==[--gtest_filter=tensor.mdata]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[tensor.mdata]=]  PROPERTIES WORKING_DIRECTORY /home/wangxuefei/workspace/DeepNetFramework/build/tests/test_tensor SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_tensor_TESTS tensor.mdata)
