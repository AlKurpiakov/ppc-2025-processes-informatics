#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"

#include <mpi.h>
#include <cstring>
#include <stdexcept>
#include "pikhotskiy_r_scatter/common/include/common.hpp"

namespace pikhotskiy_r_scatter {

PikhotskiyRScatterSEQ::PikhotskiyRScatterSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = nullptr;
}

bool PikhotskiyRScatterSEQ::IsTypeSupported(MPI_Datatype datatype) const {
  return (datatype == MPI_INT || datatype == MPI_FLOAT || datatype == MPI_DOUBLE);
}

size_t PikhotskiyRScatterSEQ::GetTypeSize(MPI_Datatype datatype) const {
  if (datatype == MPI_INT) {
    return sizeof(int);
  } else if (datatype == MPI_FLOAT) {
    return sizeof(float);
  } else if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}

bool PikhotskiyRScatterSEQ::ValidationImpl() {
  const auto& input_data = GetInput();
  int send_element_count = std::get<1>(input_data);
  MPI_Datatype send_datatype = std::get<2>(input_data);
  void* receive_data_ptr = std::get<3>(input_data);
  int receive_element_count = std::get<4>(input_data);
  MPI_Datatype receive_datatype = std::get<5>(input_data);
  int root_process = std::get<6>(input_data);
  MPI_Comm comm = std::get<7>(input_data);
  
  // Подавляем предупреждения о неиспользуемых переменных
  (void)receive_datatype;
  (void)root_process;
  (void)comm;
  
  // Проверяем корректность количества элементов
  if (send_element_count < 0 || receive_element_count < 0) {
    throw std::runtime_error("Element count must be non-negative");
  }
  
  if (send_element_count != receive_element_count) {
    throw std::runtime_error("Send and receive element counts must match");
  }
  
  // Проверяем совпадение типов данных
  if (send_datatype != receive_datatype) {
    throw std::runtime_error("Send and receive datatypes must match");
  }
  
  // Проверяем указатели на данные
  if (receive_element_count > 0 && receive_data_ptr == nullptr) {
    throw std::runtime_error("Receive buffer pointer cannot be null when count > 0");
  }
  
  // Проверяем, что тип данных поддерживается
  if (!IsTypeSupported(send_datatype)) {
    throw std::runtime_error("Unsupported MPI datatype. Only MPI_INT, MPI_FLOAT, MPI_DOUBLE are supported");
  }
  
  return true;
}

bool PikhotskiyRScatterSEQ::PreProcessingImpl() {
  return true;
}

bool PikhotskiyRScatterSEQ::RunImpl() {
  const auto& input_data = GetInput();
  const void* send_data_ptr = std::get<0>(input_data);
  int send_element_count = std::get<1>(input_data);
  MPI_Datatype send_datatype = std::get<2>(input_data);
  void* receive_data_ptr = std::get<3>(input_data);
  int receive_element_count = std::get<4>(input_data);
  MPI_Datatype receive_datatype = std::get<5>(input_data);
  int root_process = std::get<6>(input_data);
  MPI_Comm comm = std::get<7>(input_data);
  
  // Подавляем предупреждения о неиспользуемых переменных
  (void)send_element_count;
  (void)receive_datatype;
  (void)root_process;
  (void)comm;
  
  // Получаем размер типа данных в байтах
  size_t element_size = GetTypeSize(send_datatype);
  
  // Вычисляем общий размер данных для копирования
  size_t total_copy_size = static_cast<size_t>(receive_element_count) * element_size;
  
  if (total_copy_size > 0) {
    if (send_data_ptr != nullptr) {
      // Если sendbuf не nullptr, копируем данные
      std::memcpy(receive_data_ptr, send_data_ptr, total_copy_size);
    } else {
      // Если sendbuf nullptr, заполняем нулями (как заглушка)
      std::memset(receive_data_ptr, 0, total_copy_size);
    }
  }
  
  // Сохраняем указатель на результат
  GetOutput() = receive_data_ptr;
  
  return true;
}

bool PikhotskiyRScatterSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pikhotskiy_r_scatter