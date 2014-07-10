// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
	//LOG(INFO) << "killing sync memory";
	clear_data();
}

inline void SyncedMemory::to_cpu() {
	//LOG(INFO) << "to cpu " << size_ << " \n";
	switch (head_) {
	case UNINITIALIZED:
		CaffeMallocHost(&cpu_ptr_, size_);
		memset(cpu_ptr_, 0, size_);
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
		break;
	case HEAD_AT_GPU:
		if (cpu_ptr_ == NULL) {
			CaffeMallocHost(&cpu_ptr_, size_);
			own_cpu_data_ = true;
		}
		CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost));
		head_ = SYNCED;
		break;
	case HEAD_AT_CPU:
	case SYNCED:
		break;
	}
}

inline void SyncedMemory::to_gpu() {
	//LOG(INFO) << "to gpu \n";
	switch (head_) {
	case UNINITIALIZED:
		//LOG(INFO) << "uninizialized " << size_ << "\n";
		CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
		own_gpu_data_ = true;
		CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
		head_ = HEAD_AT_GPU;
		break;
	case HEAD_AT_CPU:
		//LOG(INFO) << "head at cpu " << size_ << "\n";
		if (gpu_ptr_ == NULL) {
			//LOG(INFO) << "allocating gpu memory\n";
			CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
			own_gpu_data_ = true;
		}
		CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
	case SYNCED:
		break;
	}
}

void SyncedMemory::clear_data(){
	if (cpu_ptr_ && own_cpu_data_) {
		//LOG(INFO) << "freeing cpu data size: " << size_ << "\n";
		CaffeFreeHost(cpu_ptr_);
		cpu_ptr_ = NULL;
	}
	if (gpu_ptr_ && own_gpu_data_) {
		//LOG(INFO) << "freeing gpu data size: " << size_ << "\n";
		CUDA_CHECK(cudaFree(gpu_ptr_));
		gpu_ptr_ = NULL;
	}
	head_ = UNINITIALIZED;
}

const void* SyncedMemory::cpu_data() {
	to_cpu();
	return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data, int size) {
	CHECK(data);
	//LOG(INFO) << "set cpu data in synch memory for size: " << size << "rpevious size: " << size_ << "\n";
	if (size != -1 && size_ != size){
		clear_data();
		size_ = size;
	}
	if (cpu_ptr_ && own_cpu_data_) {
		CaffeFreeHost(cpu_ptr_);
	}

	cpu_ptr_ = data;
	head_ = HEAD_AT_CPU;
	own_cpu_data_ = false;
}

void SyncedMemory::set_gpu_data(void* data, int size) {
	CHECK(data);
	//LOG(INFO) << "set gpu data in synch memory for size: " << size << "rpevious size: " << size_ << "\n";
	if (size != -1 && size_ != size){
		//LOG(INFO) << "clearing data\n";
		clear_data();
		//LOG(INFO) << "data cleared\n";
		size_ = size;
	}
	if (gpu_ptr_ && own_gpu_data_) {
		CUDA_CHECK(cudaFree(gpu_ptr_));
	}

	gpu_ptr_ = data;
	head_ = HEAD_AT_GPU;
	own_gpu_data_ = false;
	//LOG(INFO) << "done setting data in synce mem\n";
}



const void* SyncedMemory::gpu_data() {
	to_gpu();
	return (const void*)gpu_ptr_;
}

void* SyncedMemory::mutable_cpu_data() {
	to_cpu();
	head_ = HEAD_AT_CPU;
	return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_ptr_;
}


}  // namespace caffe

