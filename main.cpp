#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

// function prototypes
torch::Tensor read_data(const std::string& location);
std::vector<std::vector<double>> csv2Dvector(const std::string& inputFileName);
std::vector<double> onelinevector(const std::vector<std::vector<double>>& invector);
void print2dvec(std::vector<std::vector<double>>& vec2print);
void print1dvector(std::vector<double>& vec2print);

// own dataset class
class MyDataset : public torch::data::Dataset<MyDataset> {

  private:
    torch::Tensor states_, labels_;

  public:
    explicit MyDataset(const std::string& loc_states, const std::string& loc_labels) {
      states_ = read_data(loc_states);
      labels_ = read_data(loc_labels);
    };

    torch::data::Example<> get(size_t index) override {
      return {states_[index], labels_[index]};
    };

    torch::optional<size_t> size() const override {
      return labels_.size(0);
    };

};


int main() {

  const std::string input_loc = "/home/marcel/projects/libtorch/dataloader_test/data/input.csv";
  const std::string output_loc = "/home/marcel/projects/libtorch/dataloader_test/data/output.csv";
  torch::Tensor csv_tensor(read_data(input_loc));
  std::cout << "Input Tensor: \n" << csv_tensor[0] << '\n';
  torch::Tensor csv_output = read_data(output_loc);
  std::cout << "Output Tensor (target): \n" << csv_output[0] << '\n';
  size_t s1 = csv_output.size(0);
  std::cout << "Length of Data: " << s1 << " \n";

  size_t batch_size = 2;

  auto data_set = MyDataset(input_loc, output_loc).map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::DistributedRandomSampler>(
    std::move(data_set),
    batch_size
  );

  for (auto& batch: *data_loader) {
    auto data = batch.data;
    auto labels = batch.target;

    data = data.to(torch::kDouble);
    labels = labels.to(torch::kDouble);
    std::cout << "Data: \n" << data << '\n';
    std::cout << "Labels: \n" << labels << '\n';
  }

  return 0;
}

// read from csv and return tensor
torch::Tensor read_data(const std::string& location) {
    //read csv
    std::vector<std::vector<double>> data_vector;
    data_vector = csv2Dvector(location);
    std::cout << "Data Vector: \n"; // debug reasons remove later
    print2dvec(data_vector); // debug remove later
    std::vector<double> flat_vector;
    flat_vector.reserve(data_vector.size() * data_vector.front().size());
    flat_vector = onelinevector(data_vector);
    std::cout << "Flat Vector: \n"; //debug
    print1dvector(flat_vector); //debug
    torch::Tensor data_tensor;
    data_tensor = torch::from_blob(flat_vector.data(),
    {
        static_cast<unsigned int>(data_vector.size()),
        static_cast<unsigned int>(data_vector.front().size())
    }, at::kDouble).clone();
    return data_tensor;
}

std::vector<std::vector<double>> csv2Dvector(const std::string& inputFileName) {
  using namespace std;

  vector<vector<double>> data;
  ifstream inputFile(inputFileName);
  int l = 0;

  while (inputFile) {
    l++;
    string s;
    if (!getline(inputFile, s))
      break;
    if (s[0] != '#') {
      istringstream ss(s);
      vector<double> record;

      while (ss) {
        string line;
        if (!getline(ss, line, ','))
          break;
        try {
          record.push_back(stof(line));
        } catch (const std::invalid_argument e) {
          cout << "NaN found in file " << inputFileName << " line " << l
               << '\n';
          e.what();
        }
      }

      data.push_back(record);
    }
  }

  if (!inputFile.eof()) {
    cerr << "Could not read file " << inputFileName << '\n';
    throw invalid_argument("File not found.");
  }

  return data;
}

std::vector<double> onelinevector(const std::vector<std::vector<double>>& invector) {

  std::vector<double> v1d;
  if (invector.size() == 0)
    return v1d;
  v1d.reserve(invector.size() * invector.front().size());

  for (auto &innervector : invector) {
    v1d.insert(v1d.end(), innervector.begin(), innervector.end());
  }

  return v1d;
}

void print2dvec(std::vector<std::vector<double>> &vec2print) {
    for (const auto& i:vec2print) {
        for (const auto& j: i) {
            std::cout << j << ' ';
        }
        std::cout << '\n';
    }
}

void print1dvector(std::vector<double>& vec2print) {
    for (const auto& i:vec2print) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
}