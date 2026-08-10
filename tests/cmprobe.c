/* Throwaway probe: enumerate VkCooperativeMatrixPropertiesKHR for the truth table. */
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    VkApplicationInfo ai; VkInstanceCreateInfo ici; VkInstance inst; VkPhysicalDevice pd;
    VkResult r;
    memset(&ai,0,sizeof ai); ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName="cmprobe"; ai.apiVersion=VK_API_VERSION_1_3;
    memset(&ici,0,sizeof ici); ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo=&ai;
    r=vkCreateInstance(&ici,NULL,&inst);
    if(r!=VK_SUCCESS){fprintf(stderr,"create instance %d\n",(int)r);return 1;}
    uint32_t n=0;
    r=vkEnumeratePhysicalDevices(inst,&n,NULL);
    if(r!=VK_SUCCESS||n==0){fprintf(stderr,"no pd %d\n",(int)r);return 1;}
    VkPhysicalDevice *pds=calloc(n,sizeof* pds);
    vkEnumeratePhysicalDevices(inst,&n,pds);
    pd=pds[0];

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR gpm=
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst,"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if(!gpm){fprintf(stderr,"gpm null\n");return 1;}
    uint32_t cnt=0;
    r=gpm(pd,&cnt,NULL);
    printf("propertyCount=%u (r=%d)\n",cnt,(int)r);
    VkCooperativeMatrixPropertiesKHR *props=calloc(cnt?cnt:1,sizeof*props);
    for(uint32_t i=0;i<cnt;i++)props[i].sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    r=gpm(pd,&cnt,props);
    printf("second call r=%d, cnt=%u\n",(int)r,cnt);
    const char* tname(VkComponentTypeKHR t){switch(t){case VK_COMPONENT_TYPE_SINT8_KHR:return "sint8";case VK_COMPONENT_TYPE_UINT8_KHR:return "uint8";case VK_COMPONENT_TYPE_SINT16_KHR:return "sint16";case VK_COMPONENT_TYPE_UINT16_KHR:return "uint16";case VK_COMPONENT_TYPE_SINT32_KHR:return "sint32";case VK_COMPONENT_TYPE_UINT32_KHR:return "uint32";case VK_COMPONENT_TYPE_SINT64_KHR:return "sint64";case VK_COMPONENT_TYPE_UINT64_KHR:return "uint64";case VK_COMPONENT_TYPE_FLOAT16_KHR:return "float16";case VK_COMPONENT_TYPE_FLOAT32_KHR:return "float32";case VK_COMPONENT_TYPE_FLOAT64_KHR:return "float64";case VK_COMPONENT_TYPE_BFLOAT16_KHR:return "bfloat16";default:return "?";}}
    for(uint32_t i=0;i<cnt;i++){
      printf("[%u] %ux%ux%u  A=%s B=%s C=%s R=%s scope=%u\n",i,props[i].MSize,props[i].NSize,props[i].KSize,tname(props[i].AType),tname(props[i].BType),tname(props[i].CType),tname(props[i].ResultType),(unsigned)props[i].scope);
    }
    free(pds);free(props);
    vkDestroyInstance(inst,NULL);
    return 0;
}
