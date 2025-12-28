#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <cstddef>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "pikhotskiy_r_scatter/common/include/common.hpp"

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterMPI::PikhotskiyRScatterMPI(const InType& in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput() = nullptr;
}

bool PikhotskiyRScatterMPI::IsSupportedType(MPI_Datatype type) const {
  return (type == MPI_INT || type == MPI_FLOAT || type == MPI_DOUBLE);
}

bool PikhotskiyRScatterMPI::ValidationImpl() {
  auto& input = this->GetInput();
  MPI_Datatype sendtype = std::get<2>(input);
  MPI_Datatype recvtype = std::get<5>(input);
  
  if (!IsSupportedType(sendtype)) {
    throw std::runtime_error("Unsupported send datatype. Only MPI_INT, MPI_FLOAT, MPI_DOUBLE are supported.");
  }
  
  if (!IsSupportedType(recvtype)) {
    throw std::runtime_error("Unsupported receive datatype. Only MPI_INT, MPI_FLOAT, MPI_DOUBLE are supported.");
  }
  
  if (sendtype != recvtype) {
    throw std::runtime_error("Send and receive datatypes must match.");
  }
  
  int sendcount = std::get<1>(input);
  int recvcount = std::get<4>(input);
  
  if (sendcount < 0 || recvcount < 0) {
    throw std::runtime_error("Counts cannot be negative.");
  }
  
  return true;
}

bool PikhotskiyRScatterMPI::PreProcessingImpl() {
  return true;
}

int PikhotskiyRScatterMPI::CustomScatterInt(const void* sendbuf, int sendcount, 
                                           void* recvbuf, int recvcount, 
                                           int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  int* recv_data = static_cast<int*>(recvbuf);
  
  if (rank == root) {
    const int* send_data = static_cast<const int*>(sendbuf);
    
    // Копируем данные для корневого процесса
    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(int));
    
    // Отправляем данные остальным процессам
    for (int i = 0; i < size; ++i) {
      if (i != root) {
        MPI_Send(send_data + i * sendcount, sendcount, MPI_INT, i, 0, comm);
      }
    }
  } else {
    MPI_Recv(recv_data, recvcount, MPI_INT, root, 0, comm, MPI_STATUS_IGNORE);
  }
  
  return MPI_SUCCESS;
}

int PikhotskiyRScatterMPI::CustomScatterFloat(const void* sendbuf, int sendcount, 
                                             void* recvbuf, int recvcount, 
                                             int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  float* recv_data = static_cast<float*>(recvbuf);
  
  if (rank == root) {
    const float* send_data = static_cast<const float*>(sendbuf);
    
    // Копируем данные для корневого процесса
    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(float));
    
    // Отправляем данные остальным процессам
    for (int i = 0; i < size; ++i) {
      if (i != root) {
        MPI_Send(send_data + i * sendcount, sendcount, MPI_FLOAT, i, 0, comm);
      }
    }
  } else {
    MPI_Recv(recv_data, recvcount, MPI_FLOAT, root, 0, comm, MPI_STATUS_IGNORE);
  }
  
  return MPI_SUCCESS;
}

int PikhotskiyRScatterMPI::CustomScatterDouble(const void* sendbuf, int sendcount, 
                                              void* recvbuf, int recvcount, 
                                              int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  double* recv_data = static_cast<double*>(recvbuf);
  
  if (rank == root) {
    const double* send_data = static_cast<const double*>(sendbuf);
    
    // Копируем данные для корневого процесса
    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(double));
    
    // Отправляем данные остальным процессам
    for (int i = 0; i < size; ++i) {
      if (i != root) {
        MPI_Send(send_data + i * sendcount, sendcount, MPI_DOUBLE, i, 0, comm);
      }
    }
  } else {
    MPI_Recv(recv_data, recvcount, MPI_DOUBLE, root, 0, comm, MPI_STATUS_IGNORE);
  }
  
  return MPI_SUCCESS;
}

int PikhotskiyRScatterMPI::CustomScatter(
    const void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm) {
  
  int sendtype_size = 0;
  MPI_Type_size(sendtype, &sendtype_size);
  
  int recvtype_size = 0;
  MPI_Type_size(recvtype, &recvtype_size);
  
  if (sendcount * sendtype_size < recvcount * recvtype_size) {
    return MPI_ERR_ARG;
  }
  
  // Проверяем, что sendbuf не nullptr для корневого процесса
  int rank;
  MPI_Comm_rank(comm, &rank);
  
  if (rank == root && sendbuf == nullptr && sendcount > 0) {
    return MPI_ERR_BUFFER;
  }
  
  if (recvbuf == nullptr && recvcount > 0) {
    return MPI_ERR_BUFFER;
  }
  
  // Выбираем соответствующую функцию в зависимости от типа данных
  if (sendtype == MPI_INT) {
    return CustomScatterInt(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  } else if (sendtype == MPI_FLOAT) {
    return CustomScatterFloat(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  } else if (sendtype == MPI_DOUBLE) {
    return CustomScatterDouble(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  } else {
    return MPI_ERR_TYPE;
  }
}

bool PikhotskiyRScatterMPI::RunImpl() {
  auto& input = this->GetInput();
  
  const void* sendbuf = std::get<0>(input);
  int sendcount = std::get<1>(input);
  MPI_Datatype sendtype = std::get<2>(input);
  void* recvbuf = std::get<3>(input);
  int recvcount = std::get<4>(input);
  MPI_Datatype recvtype = std::get<5>(input);
  int root = std::get<6>(input);
  MPI_Comm comm = std::get<7>(input);
  
  int result = CustomScatter(
      sendbuf, sendcount, sendtype,
      recvbuf, recvcount, recvtype,
      root, comm
  );
  
  if (result != MPI_SUCCESS) {
    throw std::runtime_error("CustomScatter failed with MPI error code: " + 
                            std::to_string(result));
  }
  
  this->GetOutput() = recvbuf;
  return true;
}

bool PikhotskiyRScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_scatter