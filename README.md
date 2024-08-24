# DeepNetFramework

A deep learning network framework based on C++ and CUDA programming

## Run

F5 -> debug

F6 -> 选择相应task


// rm -rf * && cmake .. -DCMAKE_VERBOSE_MAKEFILE=on && make

// rm -rf * && cmake .. && make

//  cd build && rm -rf * && cmake .. -DCMAKE_VERBOSE_MAKEFILE=on -DCMAKE_BUILD_TYPE=debug && make


## Tools

[如何在 Ubuntu 22.04 LTS Jammy Jellyfish 上的多个 GCC 和 G++ 编译器版本之间切换](https://cn.linux-console.net/?p=10266#:~:text=%E9%A6%96%E5%85%88%E6%89%93%E5%BC%80%E5%91%BD%E4%BB%A4%E8%A1%8C%E7%BB%88%E7%AB%AF%E5%B9%B6%E4%BD%BF%E7%94%A8%E4%BB%A5%E4%B8%8B%20apt%20%E5%91%BD%E4%BB%A4%E5%9C%A8%20Ubuntu%2022.04%20%E4%B8%8A%E5%AE%89%E8%A3%85%E5%87%A0%E4%B8%AA%E4%B8%8D%E5%90%8C%E7%89%88%E6%9C%AC%E7%9A%84%20GCC%20%E5%92%8C,-y%20install%20gcc-8%20g%2B%2B-8%20gcc-9%20g%2B%2B-9%20gcc-10%20g%2B%2B-10)


## issues

### c++11与getes1.11编译报错

[GCC 11, operator<<  doesn't work in INSTANTIATE_TEST_SUITE_P](https://github.com/google/googletest/issues/4079)

> 解决：更新gtest版本（1.15ok）




