#include <random>
#include <gtest/gtest.h>

#include "compute_graph.h"
#include "network.h"

#include "l1loss.h"
#include "Operators.h"

#include "Op_matmul.h"
#include "Op_elementwise.h"


TEST(perceptron, matmul_WX) {
    ComputeGraph* mm_graph = new ComputeGraph();
    mm_graph->_inputTensors.push_back(new Tensor());
    mm_graph->_weightTensors.push_back(new Tensor({2, 2}));
    new Op_matmul(mm_graph->_inputTensors[0], mm_graph->_weightTensors[0]);

    Tensor *input = new Tensor({1, 2});
    Tensor *target = new Tensor({1, 2});

    Network mm_net(mm_graph, cudaStreamDefault);
    mm_net.to(cudaMemoryTypeDevice);
    mm_net._weightTensors[0]->fillDataRandom(-1.0, 1.0);
    std::vector<Tensor*> init_out = mm_net.init({input}, "");


    ComputeGraph *l1loss_graph = new L1loss();
    Network l1loss(l1loss_graph, cudaStreamDefault);
    l1loss.to(cudaMemoryTypeDevice);
    l1loss.init({init_out[0], target}, "");

    int epoches = 1;
    for (int i = 0; i < epoches; i++) {
        input->fillDataRandom(-1.0, 1.0);
        target->_pdata[0] = input->_pdata[0] + input->_pdata[1];
        target->_pdata[1] = input->_pdata[0] - input->_pdata[1];

        // std::cout << "input" << *input;
        input->to(cudaMemoryTypeDevice);
        std::vector<Tensor*> in{input};
        // mm_net._weightTensors[0]->to(cudaMemoryTypeHost);
        // input->to(cudaMemoryTypeHost);
        std::vector<Tensor*> predict = mm_net.forward(in);
        // std::cout << "target" << *target;
        // std::cout << "predict" << *predict[0];

        // predict[0]->to(cudaMemoryTypeHost);
        target->to(cudaMemoryTypeDevice);
        std::vector<Tensor*> pre_target{predict[0], target};
        std::vector<Tensor*> loss = l1loss.forward(pre_target);

        std::cout << "loss: " << *loss[0];

        l1loss.backward();
        mm_net.getOutputTensors()[0] = l1loss._inputTensors[0];
        mm_net.backward();

        mm_net.updateWeights(0.01);
    }
    std::cout << "weight" << *mm_net._weightTensors[0];
    l1loss.getOutputTensors()[0]->to(cudaMemoryTypeHost);
    EXPECT_LE(*l1loss.getOutputTensors()[0]->_pdata, 0.005);
}