/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "mori/application/transport/sdma/anvil.hpp"

#include <fstream>
#include <iostream>

namespace anvil
{

auto checkHsaError = [](hsa_status_t s, const char* msg, const char* file, int line) {
   if (s != HSA_STATUS_SUCCESS)
   {
      const char* hsa_err_msg;
      hsa_status_string(s, &hsa_err_msg);
      throw(std::runtime_error{std::string("HSA error at ") + file + std::string(":") + std::to_string(line) +
                               std::string(" - ") + hsa_err_msg});
   }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

#define CHECK_HSAKMT_SUCCESS(call, msg)                                                                                \
   do                                                                                                                  \
   {                                                                                                                   \
      if ((call) != HSAKMT_STATUS_SUCCESS)                                                                             \
      {                                                                                                                \
         std::cout << "ERROR code: " << std::dec << call << " " << msg << " (File: " << __FILE__                       \
                   << ", Line: " << __LINE__ << ")" << std::endl;                                                      \
         exit(EXIT_FAILURE);                                                                                           \
      }                                                                                                                \
   } while (0)

#if 0
inline void checkHipError(hipError_t err, const char* msg, const char* file, int line)
{
   if (err != hipSuccess)
   {
      std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
                << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
      std::exit(EXIT_FAILURE);
   }
}

#define CHECK_HIP_ERROR(cmd) checkHipError((cmd), #cmd, __FILE__, __LINE__)

// Allow access to peerDeviceId from deviceId
inline void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
   int canAccess;
   CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
   if (!canAccess)
   {
      std::cerr << "Unable to enable peer access from GPU devices " << deviceId << " to " << peerDeviceId << "\n";
   }

   CHECK_HIP_ERROR(hipSetDevice(deviceId));
   hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
   if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
   {
      std::cerr << "Unable to enable peer to peer access from " << deviceId << "  to " << peerDeviceId << " ("
                << hipGetErrorString(error) << ")\n";
   }
}
#endif

// HSA agents
std::vector<hsa_agent_t> cpuAgents_;
std::vector<hsa_agent_t> gpuAgents_;

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent, hsa_device_type_t target_device_type,
                                     [[maybe_unused]] void* vector)
{
   std::vector<hsa_agent_t>* agents = static_cast<std::vector<hsa_agent_t>*>(vector);
   hsa_device_type_t device_type{};
   hsa_status_t status{hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};
   if (status != HSA_STATUS_SUCCESS)
   {
      printf("Failure to get device type: 0x%x", status);
      return status;
   }
   if (device_type == target_device_type)
   {
      agents->push_back(agent);
   }
   return status;
}

hsa_status_t rocm_hsa_gpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context)
{
   return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_GPU, context);
}
hsa_status_t rocm_hsa_cpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context)
{
   return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_CPU, context);
}

void SetUpKFD()
{
   CHECK_HSAKMT_SUCCESS(hsaKmtOpenKFD(), "hsaKmtOpenKFD() failed!");
   HsaSystemProperties m_SystemProperties;
   memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));
   CHECK_HSAKMT_SUCCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties), "Failed!");
}

// void SetUpKFD(uint32_t targetDevice) {
//     HsaNodeProperties m_node_props;
//     CHECK_HSAKMT_SUCCESS(hsaKmtGetNodeProperties(targetDevice, &m_node_props), "Failed!");
//     std::cout << "Num of PCIe SDMA Queues: " << m_node_props.NumSdmaEngines << std::endl;
//     std::cout << "Num of XGMI SDMA Queues: " << m_node_props.NumSdmaXgmiEngines << std::endl;
//     std::cout << "Device Id: " << m_node_props.DeviceId << std::endl;
// }

void CloseKFD()
{
   CHECK_HSAKMT_SUCCESS(hsaKmtCloseKFD(), "hsaKmtCloseKFD() failed");
}

// Convert a logical deviceId index to the NVML device minor number
static const std::string getBusId(int deviceId)
{
   // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
   // format. Still need to allocate proper space in case PCI domain goes
   // higher.
   char busIdChar[] = "00000000:00:00.0";
   CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
   // we need the hex in lower case format
   for (size_t i = 0; i < sizeof(busIdChar); i++)
   {
      busIdChar[i] = std::tolower(busIdChar[i]);
   }
   return std::string(busIdChar);
}

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent, uint32_t engineId)
    : remoteDeviceId_(remoteDeviceId)
{
   // cachedWptr_(detail::gpuCallocUncachedShared<uint64_t>()),
   // committedWptr_(detail::gpuCallocUncachedShared<uint64_t>()) {
   int originalDeviceId;

   CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId)); // Save the current device
    

   uint32_t localNodeId;
   hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);
   if (status != HSA_STATUS_SUCCESS)
   {
      printf("Failure to get device info: 0x%x", status);
      //return status;
   }

   // std::cout << "Allocating queue for engine " << engineId << " on device " << localDeviceId << "
   // to device "
   //           << remoteDeviceId << std::endl;
   // std::cout << "original device id: " << originalDeviceId << " local " << localDeviceId << "
   // remote " << remoteDeviceId
   //           << " local node " << localNodeId << std::endl;

   // Allocate SDMA queue buffer on device side, requires ExecuteAccess
   HsaMemFlags memFlags = {};
   memFlags.ui32.NonPaged = 1;
   memFlags.ui32.HostAccess = 1;
   memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
   memFlags.ui32.NoNUMABind = 1;
   memFlags.ui32.ExecuteAccess = 1;
   memFlags.ui32.Uncached = 1;
   
   std::cout << "Allocating SDMA Queue Buffer for device: " << localNodeId << std::endl <<
   std::flush;


   CHECK_HSAKMT_SUCCESS(hsaKmtAllocMemory(localNodeId, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_), "Failed");
   CHECK_HSAKMT_SUCCESS(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, NULL), "Failed");

   // Create SDMA Queue
   // TODO needed here?
   memset(&queue_, 0, sizeof(HsaQueueResource));

   CHECK_HSAKMT_SUCCESS(hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID, DEFAULT_QUEUE_PERCENTAGE,
                                             DEFAULT_PRIORITY, engineId, queueBuffer_, SDMA_QUEUE_SIZE, nullptr,
                                             &queue_),
                        "Failed");

   // Populate Device Handle
   // TODO uncached
   CHECK_HIP_ERROR(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
   CHECK_HIP_ERROR(hipExtMallocWithFlags((void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
   CHECK_HIP_ERROR(hipExtMallocWithFlags((void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));

   uint64_t cachedWptr = (uint64_t) * (queue_.Queue_write_ptr_aql);
   uint64_t committedWptr = (uint64_t) * (queue_.Queue_write_ptr_aql);
   SdmaQueueDeviceHandle handle = {
       .queueBuf = static_cast<uint32_t*>(queueBuffer_),
       .rptr = queue_.Queue_read_ptr_aql,
       .wptr = queue_.Queue_write_ptr_aql,
       .doorbell = queue_.Queue_DoorBell_aql,
       .cachedWptr = cachedWptr_,
       .committedWptr = committedWptr_,
       .cachedHwReadIndex = (uint64_t) * (queue_.Queue_read_ptr_aql),
   };

   CHECK_HIP_ERROR(hipMemcpy(deviceHandle_, &handle, sizeof(SdmaQueueDeviceHandle), hipMemcpyHostToDevice));
   CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
   CHECK_HIP_ERROR(hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue()
{
   // TODO catch exception?
   CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
   CHECK_HIP_ERROR(hipFree(deviceHandle_));
   CHECK_HIP_ERROR(hipFree(cachedWptr_));
   CHECK_HIP_ERROR(hipFree(committedWptr_));
   CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
   CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE), "Failed");
}

SdmaQueueDeviceHandle* SdmaQueue::deviceHandle() const
{
   return deviceHandle_;
}

AnvilLib::~AnvilLib()
{
   for (auto& p : sdma_channels_)
   {
      p.second.clear();
   }
   CloseKFD();
   hsa_shut_down();
}

void AnvilLib::init()
{
   std::call_once(init_flag, []() {
      //   std::atexit(CloseKFD); // Register cleanup

      // HSA
      hsa_status_t status{hsa_init()};
      if (status != HSA_STATUS_SUCCESS)
      {
         printf("Failure to open HSA connection: 0x%x", status);
         //return 1;
      }
      status = hsa_iterate_agents(&rocm_hsa_gpu_agent_callback, &gpuAgents_);
      if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
      {
         printf("Failure to iterate HSA agents: 0x%x", status);
         //return 1;
      }
      status = hsa_iterate_agents(&rocm_hsa_cpu_agent_callback, &cpuAgents_);
      if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
      {
         printf("Failure to iterate HSA agents: 0x%x", status);
         //return 1;
      }

      SetUpKFD();
   });
}

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels)
{
   uint32_t engineId = getSdmaEngineId(srcDeviceId, dstDeviceId); // + 1) * 2;
   std::cout << "Connect from " << srcDeviceId << " to " << dstDeviceId << " with " << numChannels
             << " channels using engine " << engineId << std::endl;
   for (int c = 0; c < numChannels; ++c)
   {
      sdma_channels_[dstDeviceId].emplace_back(
          std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId, gpuAgents_[srcDeviceId], engineId));
   }
   return true;
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcDeviceId, int dstDeviceId, int channel_idx)
{
   if (sdma_channels_.find(dstDeviceId) == sdma_channels_.end())
   {
      return nullptr;
   }

   if (!(channel_idx < sdma_channels_[dstDeviceId].size()))
   {
      return nullptr;
   }

   return sdma_channels_[dstDeviceId][channel_idx].get(); // TODO
}

AnvilLib& AnvilLib::getInstance()
{
   static AnvilLib *instance;
   if (instance == nullptr) {
     instance = new AnvilLib();
   }
   return *instance;
}

int AnvilLib::getOamId(int deviceId)
{
   std::string busId = getBusId(deviceId);
   std::string file_str = "/sys/bus/pci/devices/" + busId + "/xgmi_physical_id";
   std::ifstream file(file_str);
   int xgmi_physical_id;
   if (file.is_open())
   {
      if (!(file >> xgmi_physical_id))
      {
         throw std::runtime_error("Failed to read xGMI physical id from file: " + file_str);
      }
   }
   else
   {
      throw std::runtime_error("Failed to open file: " + file_str);
   }
   return xgmi_physical_id;
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId)
{
   int srcOamId = getOamId(srcDeviceId);
   int dstOamId = getOamId(dstDeviceId);

   // Use even engines only
   return mi300xOamMap[srcOamId][dstOamId] * 2;
}

AnvilLib& anvil = anvil.getInstance();

} // namespace anvil
