#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "pikhotskiy_r_scatter/common/include/common.hpp"

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterMPI::PikhotskiyRScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool PikhotskiyRScatterMPI::ValidationImpl() {
  InType input = GetInput();
  MPI_Datatype sendtype = std::get<2>(input);
  MPI_Datatype recvtype = std::get<5>(input);

  if (sendtype != recvtype) {
    return false;
  }

  int sendcount = std::get<1>(input);
  int recvcount = std::get<4>(input);

  if (sendcount < 0 || recvcount < 0) {
    return false;
  }

  return true;
}

bool PikhotskiyRScatterMPI::PreProcessingImpl() {
  return true;
}

int PikhotskiyRScatterMPI::CustomScatterInt(const void *sendbuf, int sendcount, void *recvbuf, int recvcount, int root,
                                            MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int *recv_data = static_cast<int *>(recvbuf);

  if (rank == root) {
    const int *send_data = static_cast<const int *>(sendbuf);

    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(int));

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

int PikhotskiyRScatterMPI::CustomScatterFloat(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                                              int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  float *recv_data = static_cast<float *>(recvbuf);

  if (rank == root) {
    const float *send_data = static_cast<const float *>(sendbuf);

    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(float));

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

int PikhotskiyRScatterMPI::CustomScatterDouble(const void *sendbuf, int sendcount, void *recvbuf, int recvcount,
                                               int root, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  double *recv_data = static_cast<double *>(recvbuf);

  if (rank == root) {
    const double *send_data = static_cast<const double *>(sendbuf);

    std::memcpy(recv_data, send_data + rank * sendcount, sendcount * sizeof(double));

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

int PikhotskiyRScatterMPI::CustomScatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                                         int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) {
  int sendtype_size = 0;
  MPI_Type_size(sendtype, &sendtype_size);

  int recvtype_size = 0;
  MPI_Type_size(recvtype, &recvtype_size);

  if (sendcount * sendtype_size < recvcount * recvtype_size) {
    return MPI_ERR_ARG;
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == root && sendbuf == nullptr && sendcount > 0) {
    return MPI_ERR_BUFFER;
  }

  if (recvbuf == nullptr && recvcount > 0) {
    return MPI_ERR_BUFFER;
  }

  if (sendtype == MPI_INT) {
    return CustomScatterInt(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  } else if (sendtype == MPI_FLOAT) {
    return CustomScatterFloat(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  } else {
    return CustomScatterDouble(sendbuf, sendcount, recvbuf, recvcount, root, comm);
  }
}

bool PikhotskiyRScatterMPI::RunImpl() {
  auto &input = GetInput();

  const void *sendbuf = std::get<0>(input);
  int sendcount = std::get<1>(input);
  MPI_Datatype sendtype = std::get<2>(input);
  void *recvbuf = std::get<3>(input);
  int recvcount = std::get<4>(input);
  MPI_Datatype recvtype = std::get<5>(input);
  int root = std::get<6>(input);
  MPI_Comm comm = std::get<7>(input);

  int result = CustomScatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

  if (result != MPI_SUCCESS) {
    return false;
  }

  GetOutput() = recvbuf;
  return true;
}

bool PikhotskiyRScatterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_scatter
