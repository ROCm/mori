// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <mpi.h>

#include <utility>  // for std::pair

namespace mori {
namespace collective {

/**
 * TopologyDetector: Automatically detects communication topology
 *
 * Determines whether ranks are:
 * - Intra-node: All ranks on the same physical node
 * - Inter-node: Ranks distributed across multiple nodes (use network)
 */
class TopologyDetector {
 public:
  // Delete constructors and assignment operators to prevent instantiation
  TopologyDetector() = delete;
  TopologyDetector(const TopologyDetector&) = delete;
  TopologyDetector(TopologyDetector&&) = delete;
  TopologyDetector& operator=(const TopologyDetector&) = delete;
  TopologyDetector& operator=(TopologyDetector&&) = delete;
  ~TopologyDetector() = delete;

  /**
   * Initialize topology detection
   * Must be called before any other methods
   */
  static void Initialize();

  /**
   * Check if all ranks in the communicator are on the same node
   * @return true if all ranks are intra-node, false otherwise
   */
  static bool IsIntraNode();

  /**
   * Get the node ID for the current rank
   * @return Node ID (same for all ranks on the same node)
   */
  static int GetNodeId();

  /**
   * Get the local rank within the node (0-based)
   * @return Local rank within node
   */
  static int GetLocalRank();

  /**
   * Get the number of ranks on the current node
   * @return Number of ranks on local node
   */
  static int GetLocalSize();

  /**
   * Get the number of nodes
   * @return Number of distinct nodes
   */
  static int GetNodeCount();

  /**
   * Get the 2D topology dimensions (rows = nodes, cols = GPUs per node)
   * @return Pair of (num_nodes, gpus_per_node)
   */
  static std::pair<int, int> Get2DTopology();

  /**
   * Get the global PE/rank number
   * @return Global PE/rank number
   */
  static int GetMyPe();

  /**
   * Get the total PE number
   * @return Total PE number
   */
  static int GetNPes();

  /**
   * Cleanup topology detection resources
   */
  static void Finalize();

 private:
  inline static int myPe = -1;                      // global PE/rank number
  inline static int nPes = -1;                      // total PE number
  inline static int nodeId = -1;                    // node ID
  inline static int localRank = -1;                 // local rank within node
  inline static int localSize = -1;                 // number of PEs on local node
  inline static int numNodes = -1;                  // total number of nodes
  inline static int gpusPerNode = -1;               // number of GPUs per node
  inline static bool intraNode = false;             // whether all ranks are on the same node
  inline static MPI_Comm nodeComm = MPI_COMM_NULL;  // communicator for ranks on the same node
  inline static bool initialized = false;           // whether the detector has been initialized
};

}  // namespace collective
}  // namespace mori
