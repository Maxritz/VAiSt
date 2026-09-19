#include "vaist_core.h"
#include "vaist_runtime.h"
#include "vaist_tensor.h"
#include "vaist_compute.h"
#include "vaist_quant.h"
#include "vaist_graph.h"
#include "vaist_model.h"
#include "vaist_nn.h"
#include "vaist_llm.h"
#include "vaist_engine.h"
#include "vaist_ai.h"
#include "vaist_distributed.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static int check(int x,const char*n){if(!x){fprintf(stderr,"[T] FAIL:%s\n",n);return 0;}fprintf(stderr,"[T] PASS:%s\n",n);return 1;}
static VaistStatus addop(const float*a,const float*b,float*o,size_t n){return vaist_add_f32(a,b,o,n);}
int main(int argc, char **argv){
#if defined(_WIN32)
    if(argc>=2 && (argv[1] && strcmp(argv[1],"VAIST_VK_PROBE=1")==0)){
        /* Child probe process: validate Vulkan device then exit silently */
        VaistRuntime*r=NULL;
        if(vaist_runtime_create(VAIST_BACKEND_VULKAN,&r)==VAIST_OK){
            void *dev=NULL,*q=NULL;uint32_t qf=0;
            int ok = (vaist_runtime_vk_state(r,&dev,&q,&qf)==VAIST_OK && dev && q) ? 0 : 1;
            vaist_runtime_destroy(r);
            return ok;
        }
        return 1;
    }
#endif
    int ok=1; float a[4]={1,2,3,4},b[4]={4,3,2,1},o[4]; size_t used=0,count=0; uint32_t tok[8]; uint32_t id=0; char text[64];
 ok&=check(vaist_get_abi_version().major==1,"abi");
 ok&=check(vaist_add_f32(a,b,o,4)==VAIST_OK&&o[0]==5&&o[3]==5,"add");
 ok&=check(vaist_matmul_f32(a,b,o,1,4,1)==VAIST_OK&&fabsf(o[0]-20)<1e-5f,"matmul");
 ok&=check(vaist_nn_relu_f32(o,1)==VAIST_OK,"relu");
 {VaistRuntime*r=NULL;VaistRuntimeInfo ri;VaistBuffer*buf=NULL;ok&=check(vaist_runtime_create(VAIST_BACKEND_CPU,&r)==VAIST_OK,"runtime");ok&=check(vaist_runtime_info(r,&ri)==VAIST_OK&&ri.backend==VAIST_BACKEND_CPU,"runtime_info");ok&=check(vaist_buffer_create(r,32,&buf)==VAIST_OK&&vaist_buffer_size(buf)==32,"buffer");vaist_buffer_destroy(buf);vaist_runtime_destroy(r);}
 {uint64_t sh[2]={2,2};VaistTensor*t=NULL;ok&=check(vaist_tensor_create(VAIST_F32,2,sh,&t)==VAIST_OK,"tensor");ok&=check(vaist_tensor_numel(t)==4,"tensor_numel");vaist_tensor_destroy(t);}
 {unsigned char q[256];ok&=check(vaist_quantize_f32(VAIST_Q8_0,a,4,q,sizeof(q),&used)==VAIST_OK,"quant");ok&=check(vaist_dequantize_f32(VAIST_Q8_0,q,used,o,4)==VAIST_OK&&fabsf(o[2]-3)<0.05f,"dequant");}
 {VaistGraph*g=NULL;ok&=check(vaist_graph_create(&g)==VAIST_OK,"graph_create");ok&=check(vaist_graph_add_binary(g,addop,&id)==VAIST_OK,"graph_add");ok&=check(vaist_graph_compile(g)==VAIST_OK,"graph_compile");ok&=check(vaist_graph_execute(g,a,b,o,4)==VAIST_OK&&o[0]==5,"graph_execute");vaist_graph_destroy(g);}
 {VaistKVCache*k=NULL;ok&=check(vaist_kv_create(4,2,&k)==VAIST_OK,"kv_create");ok&=check(vaist_kv_write(k,0,a)==VAIST_OK&&vaist_kv_length(k)==1,"kv_write");ok&=check(vaist_kv_read(k,0,o)==VAIST_OK&&o[1]==2,"kv_read");vaist_kv_destroy(k);}
  {VaistTokenizer*t=NULL;ok&=check(vaist_tokenizer_create(VAIST_TOKENIZER_BYTES,NULL,&t)==VAIST_OK,"tok_create");ok&=check(vaist_tokenize_bytes("abc",tok,8,&count)==VAIST_OK&&count==3&&tok[0]==97,"tokenize");vaist_tokenizer_destroy(t);}
 {VaistEngine*e=NULL;VaistSession*s=NULL;ok&=check(vaist_engine_create(&e)==VAIST_OK,"engine");ok&=check(vaist_session_create(e,&s)==VAIST_OK,"session");ok&=check(vaist_generate(s,"x",3,text,sizeof(text))==VAIST_OK&&text[0]=='x',"generate");vaist_session_destroy(s);vaist_engine_destroy(e);}
 {float c;ok&=check(vaist_vector_cosine(a,a,4,&c)==VAIST_OK&&fabsf(c-1)<1e-5f,"cosine");}
 {VaistCommunicator c;ok&=check(vaist_comm_init(&c,0,1)==VAIST_OK&&vaist_allreduce_sum_f32(&c,a,4)==VAIST_OK,"distributed");}
 return ok?0:1;
}
