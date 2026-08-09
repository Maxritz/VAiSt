/**
 * \file vkdist_internal.h
 * \brief Internal vkdist structures: opcodes, frame constants, server state.
 *
 * The wire protocol constants live here (shared by the server RPC loop and
 * the client framing helpers in vkdist.c). The public API in vkdist.h stays
 * OS-agnostic; socket details are confined to vkdist.c.
 */
#ifndef VKDIST_INTERNAL_H
#define VKDIST_INTERNAL_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

#include "vkruntime.h"
#include "vkblas.h"
#include "vkdist.h"

/* ── Frame / protocol limits ────────────────────────────────────────────────
 * A frame is u32 LE length + u32 LE opcode + payload. Phase 0 caps the payload
 * at 1 MiB so the server can use one fixed-size receive buffer per connection.
 */
#define VKDIST_MAX_FRAME     ((uint32_t)(1u << 20)) /**< Max payload bytes/frame. */
#define VKDIST_LISTEN_BACKLOG 8                     /**< TCP listen backlog.      */

#define VKDIST_MAX_BUFFERS   64u                    /**< Remote buffers/connection. */

/* ── Opcodes ──────────────────────────────────────────────────────────────── *
 * Every reply echoes the request opcode and begins with an i32 VkResult
 * (0 = VK_SUCCESS). See specs/VKDIST-DESIGN.md section 4 for the wire layout.
 */
typedef enum {
    VKDIST_OP_HELLO           = 1,  /**< handshake: u32 version.            */
    VKDIST_OP_REGISTER_BUFFER = 2,  /**< alloc remote buffer.               */
    VKDIST_OP_UPLOAD          = 3,  /**< host -> remote buffer.             */
    VKDIST_OP_DISPATCH_GEMM   = 4,  /**< remote vkblas_sgemm.               */
    VKDIST_OP_READBACK        = 5,  /**< remote buffer -> host.             */
    VKDIST_OP_BYE             = 6,  /**< graceful close (no reply).         */
} vkdist_opcode_t;

/* ── Server-side remote buffer table ─────────────────────────────────────── */

typedef struct {
    uint64_t       handle;  /**< Opaque handle handed to the client.       */
    VkBuffer       buffer;  /**< Server-device buffer (vkr_malloc'd).      */
    VkDeviceMemory memory;  /**< Backing block memory (for cleanup).       */
    VkDeviceSize   size;    /**< Registered size in bytes.                 */
} vkdist_remote_buffer_t;

/* ── Per-connection server session state ───────────────────────────────────
 * One serialized request/reply loop owns one command buffer, one runtime, and
 * one buffer table. Created and destroyed entirely inside vkdist_server_run().
 */
typedef struct {
    VkPhysicalDevice pd;            /**< Physical device.                  */
    VkDevice         dev;           /**< Logical device.                   */
    VkQueue          queue;         /**< Family-0 queue (compute/transfer).*/
    VkCommandPool    cmd_pool;      /**< RESET_COMMAND_BUFFER pool.        */
    VkCommandBuffer  cmd;           /**< Shared, always-reset-between-ops. */
    VkFence          fence;         /**< Dispatch sync fence.              */
    VkRuntime       *runtime;       /**< Transient runtime for this conn.  */
    VkBLASContext   *blas;          /**< Caller-owned BLAS context.        */
    /* Serializes access to the shared VkBLASContext (which is not
       thread-safe: pipeline cache + descriptor pool) when a single context
       is shared by multiple serve_many worker threads. NULL for the
       single-connection vkdist_server_run path. Type is pthread_mutex_t*;
       kept opaque here so vkdist_internal.h stays pthread-free. */
    void            *blas_lock;
    vkdist_remote_buffer_t buffers[VKDIST_MAX_BUFFERS];
    uint32_t         buffer_count;  /**< Live remote buffers.              */
    uint64_t         next_handle;   /**< Monotonic handle allocator.       */
} vkdist_server_state_t;

#endif /* VKDIST_INTERNAL_H */
