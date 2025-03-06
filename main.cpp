#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#define INNODE 2
#define HIDENODE 4
#define OUTNODE 1

double rate = 0.8;       // learning rate
double threshold = 1e-4; // threshold for error
size_t mosttimes = 1e6;  // max times of learning

struct Sample {
    std::vector<double> in, out;
};

struct Node {
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;
};

namespace utils {
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> getFileData(std::string filename) {
        std::vector<double> res;

        std::ifstream in(filename);
        if (in.is_open()) {
            while (!in.eof()) {
                double buffer;
                in >> buffer;
                res.push_back(buffer);
            }
            in.close();
        } else {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
        }

        return res;
    }

    std::vector<Sample> getTrainData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i+= INNODE + OUTNODE) {
            Sample sample;
            for (size_t j = 0; j < INNODE; j++) {
                sample.in.push_back(buffer[i + j]);
            }
            for (size_t j = 0; j < OUTNODE; j++) {
                sample.out.push_back(buffer[i + INNODE + j]);
            }
            res.push_back(sample);
        }

        return res;
    }

    std::vector<Sample> getTestData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i+= INNODE) {
            Sample sample;
            for (size_t j = 0; j < INNODE; j++) {
                sample.in.push_back(buffer[i + j]);
            }
            res.push_back(sample);
        }

        return res;
    }
}

Node *inputLayer[INNODE], *hideLayer[HIDENODE], *outputLayer[OUTNODE];

inline void init() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i] = new Node();
        for (size_t j = 0; j < HIDENODE; j++) {
            ::inputLayer[i]->weight.push_back(distribution(rd));
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i] = new Node();
        ::hideLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outputLayer[i] = new Node();
        ::outputLayer[i]->bias = distribution(rd);
    }
}

inline void reset_delta() {

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i]->bias_delta = 0.f;
        ::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outputLayer[i]->bias_delta = 0.f;
    }
}


int main(int arge, char *argv[]) {

    init();

    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

    for (size_t times = 0; times < mosttimes; times++) {

        reset_delta();

        double error_max = 0.f;

        for (auto &idx : train_data) {
            for (size_t i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = idx.in[i];
            }

            //正向传播
            for (size_t j = 0; j < HIDENODE; j++) {
                double sum = 0.f;
                for (size_t i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }

            for (size_t j = 0; j < OUTNODE; j++) {
                double sum = 0.f;
                for (size_t i = 0; i < HIDENODE; i++) {
                    sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
                }
                sum -= ::outputLayer[j]->bias;

                ::outputLayer[j]->value = utils::sigmoid(sum);
            }

            //计算误差
            double error = 0.f;
            for (size_t i = 0; i < OUTNODE; i++) {
                double tmp = std::fabs(::outputLayer[i]->value - idx.out[i]);
                error += tmp * tmp / 2;
            }

            error_max = std::max(error_max, error);

            //反向传播
            for (size_t i = 0; i < OUTNODE; i++) {
                double bias_delta = -(idx.out[i] - ::outputLayer[i]->value) * ::outputLayer[i]->value * (1.0 - ::outputLayer[i]->value);
                ::outputLayer[i]->bias_delta += bias_delta;
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                for (size_t j = 0; j < OUTNODE; j++) {
                    double weight_delta = (idx.out[j] - ::outputLayer[i]->value) * ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) * ::hideLayer[i]->value;
                    ::hideLayer[i]->weight_delta[j] += weight_delta;
                }
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                double sum = 0.f;
                for (size_t j = 0; j < OUTNODE; j++) {
                    sum += -(idx.out[j] - :: outputLayer[j]->value) * ::outputLayer[j]->value * (1.0 - ::outputLayer[j]->value) * ::hideLayer[i]->weight[j] * ::hideLayer[i]->value * (1.0 - ::hideLayer[i]->value);
                }
                ::hideLayer[i]->bias_delta += sum;
            }

            for (size_t i = 0; i < INNODE; i++) {
                for (size_t j = 0; j < HIDENODE; j++) {
                    double sum = 0.f;
                    for (size_t k = 0; k < OUTNODE; k++) {
                        sum += (idx.out[k] - ::outputLayer[k]->value) * ::outputLayer[k]->value * (1.0 - ::outputLayer[k]->value) * ::hideLayer[j]->weight[k] * ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value) * ::inputLayer[i]->value;
                    }
                    ::inputLayer[i]->weight_delta[j] += sum;
                }
            }
        }

        if (error_max < ::threshold) {
            std::cout << "Learning finished in " << times + 1 << " times" << std::endl;
            std::cout << "Error: " << error_max << std::endl;
            break;
        }

        auto train_data_size = double(train_data.size());

        for (size_t i = 0; i < INNODE; i++) {
            for (size_t j = 0; j < HIDENODE; j++) {
                ::inputLayer[i]->weight[j] += rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDENODE; i++) {
            ::hideLayer[i]->bias += rate * hideLayer[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < OUTNODE; j++) {
                ::hideLayer[i]->weight[j] += rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < OUTNODE; i++) {
            ::outputLayer[i]->bias += rate * ::outputLayer[i]->bias_delta / train_data_size;
        }
    }

    return 0;
}