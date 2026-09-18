#include "vaist_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#ifndef DBG_TRACE
#define DBG_TRACE(...) do { fprintf(stderr, "[T] %s:%d %s: ", __FILE__, __LINE__, __func__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if VAIST_ENABLE_VULKAN && defined(VAIST_VULKAN_HDR)
#include <vulkan/vulkan.h>
#define VAIST_HAVE_VK_HDR 1
#else
#define VAIST_HAVE_VK_HDR 0
#endif

struct VaistRuntime{
    VaistRuntimeInfo info;
    void *loader;          /* vulkan-1.dll / libvulkan.so.1 (optional) */
    uint32_t have_caps;    /* cache: device caps probed */
    VaistDeviceCaps caps;
#if VAIST_HAVE_VK_HDR
    /* lazily-initialized Vulkan device state; NULL/VK_NULL_HANDLE when no device. */
    PFN_vkGetInstanceProcAddr gipa;   /* cached from the loader; used for all proc resolution */
    VkInstance instance;
    VkPhysicalDevice physdev;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkPhysicalDeviceMemoryProperties memprops;
    uint32_t vk_ready;     /* 1 once a device+queue were created */
    uint32_t vk_dev_attempted; /* 1 once vkCreateDevice was tried (success or crash-guarded failure) */
    uint32_t vk_8bit;      /* 1 if VK_KHR_8bit_storage was enabled */
#endif
};
struct VaistBuffer{
    VaistRuntime *rt;
    void *data;          /* host-mapped (Vk) or malloc'd (CPU) memory */
    size_t size;
#if VAIST_HAVE_VK_HDR
    VkBuffer vkbuf;      /* resident device buffer, or VK_NULL_HANDLE */
    VkDeviceMemory vkmem;
    uint32_t mapped;     /* 1 while vkmem is persistently mapped */
#endif
};
struct VaistStream{VaistRuntime *rt;};

static void* runtime_load_vk(const char*name){
#if defined(_WIN32)
    return (void*)LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW|RTLD_LOCAL);
#endif
}
static void* runtime_sym(void*h,const char*name){
#if defined(_WIN32)
    return h?(void*)GetProcAddress((HMODULE)h,name):NULL;
#else
    return h?dlsym(h,name):NULL;
#endif
}
static void runtime_close(void*h){
#if defined(_WIN32)
    if(h) FreeLibrary((HMODULE)h);
#else
    if(h) dlclose(h);
#endif
}

#if VAIST_HAVE_VK_HDR && defined(_WIN32)
/* The sandbox loader (and some fake-ICDs) accept vkCreateInstance and enumerate
 * a phantom physical device, but vkCreateDevice crashes the whole process
 * (stack overflow / access violation) instead of returning an error. We cannot
 * catch that in-process (no /EHa, and STACK_OVERFLOW is non-continuable), so we
 * test device-creation viability in a child process with its own stack. If the
 * child crashes or exits non-zero, there is no usable GPU and callers fall back
 * to the scalar CPU kernels (the vaist_blas weak-box contract).
 *
 * The child is this same executable, launched with VAIST_VK_PROBE=1 in its
 * environment; the test binary's main() honours that as a probe-only entry. */
/* Probe verdict is machine state: spawn at most one child per process. Every
 * spawn walks driver-hooked process creation, so repeated spawns multiply
 * exposure to faulty ICD hooks. (Concurrent first calls may race one extra
 * spawn; harmless — same verdict.) */
static int g_probe_done=0;
static int g_probe_ok=0;
static int vaist_vk_device_probe_child(void){
    wchar_t wself[MAX_PATH];
    wchar_t cmd[MAX_PATH*2];
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    DWORD code=1;
    UINT prevMode;
    DWORD nself;
    DBG_TRACE("probe-child enter");
    nself=GetModuleFileNameW(NULL,wself,MAX_PATH);
    if(nself==0||nself>=MAX_PATH){DBG_TRACE("probe-child path=self-name-bad len=%lu -> 0",(unsigned long)nself);return 0;}
    /* Pass the probe flag as a COMMAND-LINE ARGUMENT (SetEnvironmentVariableA does
     * NOT propagate to CreateProcessW's inherited env block on Windows, so an
     * env var would be invisible to the child and every child would re-spawn →
     * runaway recursion). The child's main() reads argv for the flag. */
    _snwprintf_s(cmd,(sizeof(cmd)/sizeof(cmd[0])),_TRUNCATE,L"%ls VAIST_VK_PROBE=1",wself);
    DBG_TRACE("probe-child cmd-built");
    memset(&si,0,sizeof(si)); si.cb=sizeof(si);
    memset(&pi,0,sizeof(pi));
    DBG_TRACE("probe-child spawning");
    /* The disposable probe child inherits the error mode: suppress system
     * crash dialogs there (a driver fault must surface only as an exit
     * code). Restored in the parent immediately after the spawn. */
    prevMode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if(!CreateProcessW(wself,cmd,NULL,NULL,FALSE,0,NULL,NULL,&si,&pi)){SetErrorMode(prevMode);DBG_TRACE("probe-child path=spawn-fail err=%lu -> 0",(unsigned long)GetLastError());return 0;}
    SetErrorMode(prevMode);
    DBG_TRACE("probe-child spawned, waiting");
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess,15000);
    if(!GetExitCodeProcess(pi.hProcess,&code)) code=1;
    CloseHandle(pi.hProcess);
    DBG_TRACE("probe-child exit-code=%lu -> %d",(unsigned long)code,(code==0)?1:0);
    return (code==0)?1:0;
}
#endif

VAIST_API VaistStatus vaist_runtime_create(VaistBackend requested, VaistRuntime **out){
    VaistRuntime *r; uint32_t has=0; void *h=NULL;
    if(!out) return VAIST_INVALID_ARGUMENT;
    *out=NULL;
    if(requested!=VAIST_BACKEND_AUTO&&requested!=VAIST_BACKEND_CPU&&requested!=VAIST_BACKEND_VULKAN)
        return VAIST_INVALID_ARGUMENT;
    if(requested==VAIST_BACKEND_CPU){
        /* CPU backend never touches Vulkan: do not load the graphics driver
         * into this process (a faulty ICD faults the whole process from its
         * own threads) and skip the child-process probe entirely. */
        DBG_TRACE("path=cpu-no-vk -> skip loader+probe");
    } else {
#if VAIST_HAVE_VK_HDR && defined(_WIN32)
    /* Probe FIRST in a disposable child (which loads the driver itself);
     * this process touches the ICD only after the child proves a usable
     * device — a faulty driver can then never fault us at load time.
     * Skip the spawn when WE are the probe child (flag on the command
     * line is reliable across the DLL/exe boundary); the vk_state/device
     * attempt below then reports viability via the exit code. */
    { wchar_t *cl=GetCommandLineW();
      int probe=cl? (wcsstr(cl,L"VAIST_VK_PROBE=1")!=NULL) : 0;
      DBG_TRACE("probe-check probe=%d",probe);
      if(!probe){
          if(!g_probe_done){ g_probe_ok=vaist_vk_device_probe_child(); g_probe_done=1; }
          else { DBG_TRACE("probe-cached ok=%d",g_probe_ok); }
          has=g_probe_ok?1u:0u;
      } else {
          has=1u;
      } }
    DBG_TRACE("probe-done has=%u",(unsigned)has);
    if(has){
#if defined(_WIN32)
    h=runtime_load_vk("vulkan-1.dll");
#else
    h=runtime_load_vk("libvulkan.so.1");
#endif
    has = h ? 1u : 0u;
    }
#else
#if defined(_WIN32)
    h=runtime_load_vk("vulkan-1.dll");
#else
    h=runtime_load_vk("libvulkan.so.1");
#endif
    has = h ? 1u : 0u;
#endif
    DBG_TRACE("runtime_create requested=%d loader=%p has=%u",(int)requested,h,(unsigned)has);
    }
    if(requested==VAIST_BACKEND_VULKAN && !has) return VAIST_DEVICE_ERROR;
    r=(VaistRuntime*)calloc(1,sizeof(*r));
    if(!r){ runtime_close(h); return VAIST_OUT_OF_MEMORY; }
    r->loader=h;
    r->info.vulkan_available=has;
#if VAIST_HAVE_VK_HDR
    r->info.vulkan_api_version = has ? VK_API_VERSION_1_4 : 0u; /* 0 if no Vulkan */
#else
    r->info.vulkan_api_version = 0u;
#endif
    r->info.backend = (requested==VAIST_BACKEND_AUTO) ? (has?VAIST_BACKEND_VULKAN:VAIST_BACKEND_CPU) : requested;
    *out=r;
    return VAIST_OK;
}

#if VAIST_HAVE_VK_HDR
/* Resolve an instance-level proc through the loader (NOT via raw GetProcAddress on
 * the DLL): the loader's global trampoline must be obtained through
 * vkGetInstanceProcAddr so it is dispatched into the active ICD correctly.
 * (Resolving instance/device procs via GetProcAddress on vulkan-1.dll directly
 *  returns non-ICD-dispatched stubs that crash on write-back enumerations on
 *  some loader+ICD combinations.) */
static void* inst_proc(VaistRuntime*r, const char*name){
    if(!r->gipa) return NULL;
    return (void*)r->gipa(r->instance, name);
}
/* Lazily create (and cache) a Vulkan instance. Returns 1 on success. */
#if defined(_MSC_VER)
#  include <excpt.h>
#  define VAIST_TRY __try
#  define VAIST_EXCEPT __except(EXCEPTION_EXECUTE_HANDLER)
#else
#  define VAIST_TRY
#  define VAIST_EXCEPT if(0)
#endif
static int vaist_ensure_instance(VaistRuntime*r){
    PFN_vkCreateInstance pCreate;
    if(r->instance) return 1;
    if(!r->loader) return 0;
    /* vkGetInstanceProcAddr is the loader's single global export that returns
     * dispatchable entry points. vkCreateInstance itself is a loader global
     * (obtained directly from the DLL), per the Vulkan loader/ICD spec. */
    r->gipa=(PFN_vkGetInstanceProcAddr)runtime_sym(r->loader,"vkGetInstanceProcAddr");
    if(!r->gipa) return 0;           /* no loader trampoline => no ICD dispatch */
    pCreate=(PFN_vkCreateInstance)runtime_sym(r->loader,"vkCreateInstance");
    if(!pCreate) return 0;
    {
        VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,NULL,"vaist",1,"vaist",1,VK_API_VERSION_1_4};
        VkInstanceCreateInfo ci={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,NULL,0,&ai,0,NULL,0,NULL};
        VkResult res=pCreate(&ci,NULL,&r->instance);
        if(res!=VK_SUCCESS){ r->instance=VK_NULL_HANDLE; return 0; }
    }
    return r->instance!=VK_NULL_HANDLE;
}

/* Lazily create (and cache) a Vulkan device + compute queue. Returns 1 on success.
 * Graceful: any failure leaves vk_ready==0 so callers fall back to CPU. */
static int vaist_ensure_device(VaistRuntime*r){
    float pri=1.0f;
    if(r->vk_ready) return 1;
    /* Do not retry vkCreateDevice: a non-conformant/fake loader may crash
     * (stack overflow / access violation) on the phony physical device, and
     * re-entering here would re-trigger the fatal fault. Once we've tried and
     * it failed (including via the SEH backstop below), report not-ready so the
     * BLAS layer falls back to the CPU kernels. */
    if(r->vk_dev_attempted) return 0;
    r->vk_dev_attempted=1;
    if(!vaist_ensure_instance(r)) return 0;
    {
        PFN_vkEnumeratePhysicalDevices enumpd=(PFN_vkEnumeratePhysicalDevices)(void*)inst_proc(r,"vkEnumeratePhysicalDevices");
        PFN_vkGetPhysicalDeviceQueueFamilyProperties qfprop=(PFN_vkGetPhysicalDeviceQueueFamilyProperties)(void*)inst_proc(r,"vkGetPhysicalDeviceQueueFamilyProperties");
        PFN_vkGetPhysicalDeviceMemoryProperties memprop=(PFN_vkGetPhysicalDeviceMemoryProperties)(void*)inst_proc(r,"vkGetPhysicalDeviceMemoryProperties");
        PFN_vkEnumerateDeviceExtensionProperties enumext=(PFN_vkEnumerateDeviceExtensionProperties)(void*)inst_proc(r,"vkEnumerateDeviceExtensionProperties");
        PFN_vkCreateDevice createDev=(PFN_vkCreateDevice)(void*)inst_proc(r,"vkCreateDevice");
        PFN_vkGetDeviceQueue getq=(PFN_vkGetDeviceQueue)(void*)inst_proc(r,"vkGetDeviceQueue");
        uint32_t n=0,i;
        if(!enumpd||!qfprop||!memprop||!enumext||!createDev||!getq) return 0;
        if(enumpd(r->instance,&n,NULL)!=VK_SUCCESS) return 0;
        if(n==0) return 0;
        if(enumpd(r->instance,&n,&r->physdev)!=VK_SUCCESS) return 0;
        /* first queue family with compute */
        qfprop(r->physdev,&n,NULL);
        {
            VkQueueFamilyProperties *qps=(VkQueueFamilyProperties*)malloc(n?n:1);
            uint32_t qf=(uint32_t)-1;
            if(!qps) return 0;
            qfprop(r->physdev,&n,qps);
            for(i=0;i<n;i++){ if((qps[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&(qps[i].queueCount>0)){ qf=i; break; } }
            free(qps);
            if(qf==(uint32_t)-1) return 0;
            r->queue_family=qf;
        }
        memprop(r->physdev,&r->memprops);
        /* enable VK_KHR_8bit_storage on the device only if the physical device
         * advertises it (matvec shaders need int8 storage). gemm needs nothing. */
        n=0; enumext(r->physdev,NULL,&n,NULL);
        if(n){
            VkExtensionProperties *ep=(VkExtensionProperties*)malloc(n*sizeof(*ep));
            if(ep){
                enumext(r->physdev,NULL,&n,ep);
                for(i=0;i<n;i++){
                    if(strcmp(ep[i].extensionName, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)==0){ r->vk_8bit=1; break; }
                }
                free(ep);
            }
        }
        {
            VkPhysicalDeviceVulkan12Features f12;
            VkPhysicalDevice8BitStorageFeatures f8;
            const char *exts[2];
            uint32_t next=0;
            VkDeviceQueueCreateInfo qi;
            VkDeviceCreateInfo dc;
            VkResult res;
            memset(&f12,0,sizeof(f12)); f12.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            memset(&f8,0,sizeof(f8)); f8.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
            qi.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; qi.pNext=NULL; qi.flags=0;
            qi.queueFamilyIndex=r->queue_family; qi.queueCount=1; qi.pQueuePriorities=&pri;
            f12.shaderInt8=VK_TRUE;
            f8.storageBuffer8BitAccess=VK_TRUE;
            f8.uniformAndStorageBuffer8BitAccess=VK_TRUE;
            if(r->vk_8bit){ f8.pNext=(void*)&f12; exts[next++]=VK_KHR_8BIT_STORAGE_EXTENSION_NAME; }
            dc.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            dc.pNext=r->vk_8bit?(void*)&f8:NULL;
            dc.flags=0;
            dc.queueCreateInfoCount=1; dc.pQueueCreateInfos=&qi;
            dc.enabledLayerCount=0; dc.ppEnabledLayerNames=NULL;
            dc.enabledExtensionCount=next; dc.ppEnabledExtensionNames=next?exts:NULL;
            res=VK_ERROR_UNKNOWN;
            {
                VAIST_TRY {
                    res=createDev(r->physdev,&dc,NULL,&r->device);
                } VAIST_EXCEPT {
                    res=VK_ERROR_DEVICE_LOST;
                    r->device=VK_NULL_HANDLE;
                }
            }
            if(res!=VK_SUCCESS){ r->device=VK_NULL_HANDLE; return 0; }
            getq(r->device,r->queue_family,0,&r->queue);
            r->vk_ready=1;
        }
    }
    return r->vk_ready;
}
#endif

static void vaist_runtime_destroy_vk(VaistRuntime*r){
    (void)r;
#if VAIST_HAVE_VK_HDR
    DBG_TRACE("enter device=%p instance=%p gipa=%p loader=%p",(void*)r->device,(void*)r->instance,(void*)r->gipa,r->loader);
    if(r->device){
        PFN_vkDestroyDevice ddev=NULL;
        if(r->gipa && r->instance)
            ddev=(PFN_vkDestroyDevice)(void*)r->gipa(r->instance,"vkDestroyDevice");
        else
            DBG_TRACE("path=device-no-gipa -> loader-sym fallback");
        if(!ddev)
            ddev=(PFN_vkDestroyDevice)runtime_sym(r->loader,"vkDestroyDevice");
        if(ddev)
            ddev(r->device,NULL);
        r->device=VK_NULL_HANDLE;
    } else {
        DBG_TRACE("path=no-device -> skip");
    }
    if(r->instance){
        PFN_vkDestroyInstance dinst=NULL;
        if(r->gipa && r->instance)
            dinst=(PFN_vkDestroyInstance)(void*)r->gipa(r->instance,"vkDestroyInstance");
        else
            DBG_TRACE("path=instance-no-gipa -> loader-sym fallback");
        if(!dinst)
            dinst=(PFN_vkDestroyInstance)runtime_sym(r->loader,"vkDestroyInstance");
        if(dinst)
            dinst(r->instance,NULL);
        r->instance=VK_NULL_HANDLE;
    } else {
        DBG_TRACE("path=no-instance -> skip");
    }
    r->vk_ready=0; r->vk_8bit=0;
#endif
}

VAIST_API void vaist_runtime_destroy(VaistRuntime *r){
    if(!r) return;
    vaist_runtime_destroy_vk(r);
    runtime_close(r->loader);
    free(r);
}
VAIST_API VaistStatus vaist_runtime_info(const VaistRuntime *r, VaistRuntimeInfo *o){
    if(!r||!o) return VAIST_INVALID_ARGUMENT;
    *o=r->info;
    return VAIST_OK;
}

#if VAIST_HAVE_VK_HDR
/* Query real device caps by walking the pNext chain. No libvulkan link:
 * resolve core Vulkan 1.x entrypoints through the loader handle.
 * Chain: Properties2 -> ShaderIntegerDotProductProperties -> Vulkan13Properties.
 * (PortabilitySubsetPropertiesKHR and the legacy SubgroupProperties struct are
 *  intentionally omitted: the fields we report live in the Vulkan 1.3/1.4 core
 *  properties structs, which are always populated by a 1.4 driver.) */
static void probe_device(VaistRuntime*r){
    VaistDeviceCaps*caps=&r->caps;
    memset(caps,0,sizeof(*caps));
    if(!r->loader) return;
    if(!vaist_ensure_instance(r)){ r->have_caps=1; return; }
    PFN_vkEnumeratePhysicalDevices en=(PFN_vkEnumeratePhysicalDevices)(void*)inst_proc(r,"vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties2 pp2=(PFN_vkGetPhysicalDeviceProperties2)(void*)inst_proc(r,"vkGetPhysicalDeviceProperties2");
    if(!(en&&pp2)){ r->have_caps=1; return; }
    {
        VkPhysicalDevice dev;
        uint32_t n=0; en(r->instance,&n,NULL);
        if(n>0 && en(r->instance,&n,&dev)==VK_SUCCESS && dev){
            VkPhysicalDeviceShaderIntegerDotProductProperties dot;
            VkPhysicalDeviceVulkan13Properties v13;
            VkPhysicalDeviceProperties2 p2;
            memset(&dot,0,sizeof(dot)); dot.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
            memset(&v13,0,sizeof(v13)); v13.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
            /* chain order: p2 -> dot -> v13 (most-derived last) */
            dot.pNext=(void*)&v13;
            memset(&p2,0,sizeof(p2)); p2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext=(void*)&dot;
            pp2(dev,&p2);
            caps->vulkan_api_version = p2.properties.apiVersion;
            caps->vendor_id = p2.properties.vendorID;
            caps->device_id = p2.properties.deviceID;
            caps->subgroup_size = v13.minSubgroupSize ? v13.minSubgroupSize : v13.maxSubgroupSize;
            if(!caps->subgroup_size) caps->subgroup_size = 32u;
            caps->max_compute_workgroup_size[0]=p2.properties.limits.maxComputeWorkGroupSize[0];
            caps->max_compute_workgroup_size[1]=p2.properties.limits.maxComputeWorkGroupSize[1];
            caps->max_compute_workgroup_size[2]=p2.properties.limits.maxComputeWorkGroupSize[2];
            caps->max_compute_workgroups=p2.properties.limits.maxComputeWorkGroupCount[0];
            caps->max_shared_mem_bytes=p2.properties.limits.maxComputeSharedMemorySize;
            caps->integer_dot_product_8bit_supported =
                (dot.integerDotProduct8BitSignedAccelerated ||
                 dot.integerDotProduct8BitUnsignedAccelerated) ? 1u : 0u;
            caps->integer_dot_product_8bit_accelerated =
                dot.integerDotProduct8BitSignedAccelerated ? 1u : 0u;
            /* 4-bit dot fields are not present in 1.4.357 headers; default 0
             * (we only use the 8-bit INT8 ternary path). */
            caps->integer_dot_product_4bit_supported = 0;
            caps->integer_dot_product_4bit_accelerated = 0;
            caps->cooperative_matrix_supported = 0; /* not queried: unreliable on RDNA2/3 */
        }
    }
    r->have_caps=1;
}
#endif

VAIST_API VaistStatus vaist_runtime_device_caps(const VaistRuntime *r, VaistDeviceCaps *o){
    if(!r||!o) return VAIST_INVALID_ARGUMENT;
#if VAIST_HAVE_VK_HDR
    if(!r->have_caps && r->info.vulkan_available){
        probe_device((VaistRuntime*)r);
    }
    *o=r->caps;
    if(!r->info.vulkan_available){ o->vulkan_api_version=0; return VAIST_DEVICE_ERROR; }
    return VAIST_OK;
#else
    *o=(VaistDeviceCaps){0};
    return VAIST_UNSUPPORTED;
#endif
}

#if VAIST_HAVE_VK_HDR
/* Allocate a host-visible, coherent VkBuffer; persistently map it so the
 * existing upload/download (memcpy on b->data) path stays identical. */
static VaistStatus vaist_vkbuf_create(VaistRuntime*r, VaistBuffer*b, size_t n){
    PFN_vkCreateBuffer pCreateBuffer=(PFN_vkCreateBuffer)runtime_sym(r->loader,"vkCreateBuffer");
    PFN_vkAllocateMemory pAllocMem=(PFN_vkAllocateMemory)runtime_sym(r->loader,"vkAllocateMemory");
    PFN_vkBindBufferMemory pBindBufferMemory=(PFN_vkBindBufferMemory)runtime_sym(r->loader,"vkBindBufferMemory");
    PFN_vkMapMemory pMapMemory=(PFN_vkMapMemory)runtime_sym(r->loader,"vkMapMemory");
    PFN_vkGetBufferMemoryRequirements pMr=(PFN_vkGetBufferMemoryRequirements)runtime_sym(r->loader,"vkGetBufferMemoryRequirements");
    VkBufferCreateInfo bc={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,n,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_SHARING_MODE_EXCLUSIVE,0,NULL};
    VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,NULL,n,0};
    VkMemoryRequirements mr;
    uint32_t t,i;
    VkResult res;
    if(!pCreateBuffer||!pAllocMem||!pBindBufferMemory||!pMapMemory||!pMr) return VAIST_INTERNAL_ERROR;
    res=pCreateBuffer(r->device,&bc,NULL,&b->vkbuf);
    if(res!=VK_SUCCESS) return VAIST_DEVICE_ERROR;
    pMr(r->device,b->vkbuf,&mr);
    /* find a HOST_VISIBLE|HOST_COHERENT memory type */
    t=(uint32_t)-1;
    for(i=0;i<r->memprops.memoryTypeCount;i++){
        if((mr.memoryTypeBits&(1u<<i)) &&
           (r->memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
           (r->memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            t=i; break;
        }
    }
    if(t==(uint32_t)-1){
        PFN_vkDestroyBuffer pDestroyBuffer=(PFN_vkDestroyBuffer)runtime_sym(r->loader,"vkDestroyBuffer");
        if(pDestroyBuffer) pDestroyBuffer(r->device,b->vkbuf,NULL);
        b->vkbuf=VK_NULL_HANDLE;
        return VAIST_DEVICE_ERROR;
    }
    ai.allocationSize=mr.size; ai.memoryTypeIndex=t;
    res=pAllocMem(r->device,&ai,NULL,&b->vkmem);
    if(res!=VK_SUCCESS){ b->vkbuf=VK_NULL_HANDLE; return VAIST_DEVICE_ERROR; }
    if(pBindBufferMemory(r->device,b->vkbuf,b->vkmem,0)!=VK_SUCCESS){
        PFN_vkFreeMemory pFreeMem=(PFN_vkFreeMemory)runtime_sym(r->loader,"vkFreeMemory");
        PFN_vkDestroyBuffer pDestroyBuffer=(PFN_vkDestroyBuffer)runtime_sym(r->loader,"vkDestroyBuffer");
        if(pFreeMem) pFreeMem(r->device,b->vkmem,NULL);
        if(pDestroyBuffer) pDestroyBuffer(r->device,b->vkbuf,NULL);
        b->vkbuf=VK_NULL_HANDLE; b->vkmem=VK_NULL_HANDLE;
        return VAIST_DEVICE_ERROR;
    }
    res=pMapMemory(r->device,b->vkmem,0,n,0,&b->data);
    if(res!=VK_SUCCESS){
        PFN_vkFreeMemory pFreeMem=(PFN_vkFreeMemory)runtime_sym(r->loader,"vkFreeMemory");
        PFN_vkDestroyBuffer pDestroyBuffer=(PFN_vkDestroyBuffer)runtime_sym(r->loader,"vkDestroyBuffer");
        if(pFreeMem) pFreeMem(r->device,b->vkmem,NULL);
        if(pDestroyBuffer) pDestroyBuffer(r->device,b->vkbuf,NULL);
        b->vkbuf=VK_NULL_HANDLE; b->vkmem=VK_NULL_HANDLE; b->data=NULL;
        return VAIST_DEVICE_ERROR;
    }
    b->mapped=1;
    return VAIST_OK;
}
#endif

VAIST_API VaistStatus vaist_buffer_create(VaistRuntime *r,size_t n,VaistBuffer **out){
    VaistBuffer*b;
    if(!r||!out||n==0) return VAIST_INVALID_ARGUMENT;
    *out=NULL;
    b=(VaistBuffer*)calloc(1,sizeof(*b));
    if(!b) return VAIST_OUT_OF_MEMORY;
#if VAIST_HAVE_VK_HDR
    if(r->info.backend==VAIST_BACKEND_VULKAN){
        if(vaist_ensure_device(r) && vaist_vkbuf_create(r,b,n)==VAIST_OK){
            b->rt=r; b->size=n; r->info.allocated_bytes+=n; *out=b; return VAIST_OK;
        }
        /* device unavailable / allocation failed: fall through to host malloc */
    }
#endif
    b->data=malloc(n);
    if(!b->data){free(b);return VAIST_OUT_OF_MEMORY;}
    b->rt=r; b->size=n;
    r->info.allocated_bytes+=n;
    *out=b; return VAIST_OK;
}
VAIST_API void vaist_buffer_destroy(VaistBuffer*b){
    if(!b) return;
    if(b->rt && b->rt->info.allocated_bytes>=b->size) b->rt->info.allocated_bytes-=b->size;
#if VAIST_HAVE_VK_HDR
    if(b->vkmem){
        VaistRuntime*r=b->rt;
        if(r && r->loader){
            PFN_vkUnmapMemory pUnmap=(PFN_vkUnmapMemory)runtime_sym(r->loader,"vkUnmapMemory");
            PFN_vkFreeMemory pFreeMem=(PFN_vkFreeMemory)runtime_sym(r->loader,"vkFreeMemory");
            PFN_vkDestroyBuffer pDestroyBuffer=(PFN_vkDestroyBuffer)runtime_sym(r->loader,"vkDestroyBuffer");
            if(b->mapped && pUnmap) pUnmap(r->device,b->vkmem);
            if(pFreeMem) pFreeMem(r->device,b->vkmem,NULL);
            if(pDestroyBuffer) pDestroyBuffer(r->device,b->vkbuf,NULL);
        }
        b->vkbuf=VK_NULL_HANDLE; b->vkmem=VK_NULL_HANDLE; b->mapped=0;
        return;
    }
    /* CPU buffer: free malloc'd data (b->data for host path) */
    free(b->data); free(b); return;
#else
    free(b->data); free(b);
#endif
}
VAIST_API void *vaist_buffer_data(VaistBuffer*b){return b?b->data:NULL;}
VAIST_API size_t vaist_buffer_size(const VaistBuffer*b){return b?b->size:0;}
VAIST_API VaistStatus vaist_buffer_upload(VaistBuffer*b,const void*src,size_t n){
    if(!b||!src||n>b->size) return VAIST_INVALID_ARGUMENT;
    memcpy(b->data,src,n);
    return VAIST_OK;
}
VAIST_API VaistStatus vaist_buffer_download(VaistBuffer*b,void*dst,size_t n){
    if(!b||!dst||n>b->size) return VAIST_INVALID_ARGUMENT;
    memcpy(dst,b->data,n);
    return VAIST_OK;
}

/* ---- additive accessors (always present; no-op stubs when no Vulkan) ---- */
VAIST_API VaistStatus vaist_buffer_gpu_handle(VaistBuffer*b,void**handle){
    if(!b||!handle) return VAIST_INVALID_ARGUMENT;
    *handle=NULL;
#if VAIST_HAVE_VK_HDR
    if(b->rt && b->rt->info.backend==VAIST_BACKEND_VULKAN && b->vkbuf){
        *handle=(void*)b->vkbuf; return VAIST_OK;
    }
    return VAIST_UNSUPPORTED;
#else
    (void)b;
    return VAIST_UNSUPPORTED;
#endif
}
VAIST_API VaistStatus vaist_runtime_vk_state(const VaistRuntime*rt,
    void**device,void**queue,uint32_t*queue_family){
    if(!rt||!device||!queue||!queue_family) return VAIST_INVALID_ARGUMENT;
    *device=NULL; *queue=NULL; *queue_family=0;
#if VAIST_HAVE_VK_HDR
    if(!rt->vk_ready && rt->info.vulkan_available==1u){
        vaist_ensure_device((VaistRuntime*)rt);
    }
    if(rt->vk_ready){
        *device=(void*)rt->device; *queue=(void*)rt->queue; *queue_family=rt->queue_family;
        return VAIST_OK;
    }
    return VAIST_UNSUPPORTED;
#else
    return VAIST_UNSUPPORTED;
#endif
}
VAIST_API void* vaist_runtime_vk_proc(const VaistRuntime*rt,const char*name){
    if(!rt||!name) return NULL;
#if VAIST_HAVE_VK_HDR
    /* Prefer the loader-spec chain: device procs via vkGetDeviceProcAddr,
     * then instance procs via vkGetInstanceProcAddr, then raw loader symbol.
     * Raw GetProcAddress on vulkan-1.dll must NOT be used for ICD-dispatched
     * procs — it returns non-dispatched stubs that crash against a real ICD. */
    if(rt->device && rt->gipa){
        PFN_vkGetDeviceProcAddr gdp=(PFN_vkGetDeviceProcAddr)rt->gipa(rt->instance,"vkGetDeviceProcAddr");
        if(gdp){ void*p=(void*)gdp(rt->device,name); if(p) return p; }
    }
    if(rt->instance && rt->gipa){ void*p=(void*)rt->gipa(rt->instance,name); if(p) return p; }
    return runtime_sym(rt->loader,name);
#else
    return NULL;
#endif
}

VAIST_API VaistStatus vaist_stream_create(VaistRuntime*r,VaistStream**out){
    VaistStream*s; if(!r||!out) return VAIST_INVALID_ARGUMENT;
    s=(VaistStream*)calloc(1,sizeof(*s)); if(!s) return VAIST_OUT_OF_MEMORY;
    s->rt=r; *out=s; return VAIST_OK;
}
VAIST_API void vaist_stream_destroy(VaistStream*s){free(s);}
VAIST_API VaistStatus vaist_stream_synchronize(VaistStream*s){
    return s?VAIST_OK:VAIST_INVALID_ARGUMENT;
}
