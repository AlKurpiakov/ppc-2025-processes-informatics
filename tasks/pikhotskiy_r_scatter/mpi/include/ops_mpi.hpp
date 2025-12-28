#pragma once

#include <mpi.h>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PikhotskiyRScatterMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Функция scatter для int
  int CustomScatterInt(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, 
                       int root, MPI_Comm comm);
  
  // Функция scatter для float
  int CustomScatterFloat(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, 
                         int root, MPI_Comm comm);
  
  // Функция scatter для double
  int CustomScatterDouble(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, 
                          int root, MPI_Comm comm);
  
  // Основная функция scatter
  int CustomScatter(
      const void* sendbuf, int sendcount, MPI_Datatype sendtype,
      void* recvbuf, int recvcount, MPI_Datatype recvtype,
      int root, MPI_Comm comm);
  
  // Проверяет, что тип данных поддерживается
  bool IsSupportedType(MPI_Datatype type) const;
};

}  // namespace pikhotskiy_r_scatter