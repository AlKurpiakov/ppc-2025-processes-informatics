#pragma once

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit PikhotskiyRScatterSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  size_t GetTypeSize(MPI_Datatype datatype) const;
};

}  // namespace pikhotskiy_r_scatter
