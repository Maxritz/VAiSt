#include "vaist_nn.h"
#include <math.h>
static int ok(float*x,size_t n){return x&&n;}
VAIST_API VaistStatus vaist_nn_relu_f32(float*x,size_t n){size_t i;if(!ok(x,n))return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)if(x[i]<0)x[i]=0;return VAIST_OK;}
VAIST_API VaistStatus vaist_nn_gelu_f32(float*x,size_t n){size_t i;const float c=0.7978845608f;if(!ok(x,n))return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++){float z=x[i];x[i]=0.5f*z*(1.0f+tanhf(c*(z+0.044715f*z*z*z)));}return VAIST_OK;}
VAIST_API VaistStatus vaist_nn_silu_f32(float*x,size_t n){size_t i;if(!ok(x,n))return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)x[i]=x[i]/(1.0f+expf(-x[i]));return VAIST_OK;}
VAIST_API VaistStatus vaist_nn_rmsnorm_f32(float*x,size_t n,float e){size_t i;float s=0;if(!ok(x,n)||e<=0)return VAIST_INVALID_ARGUMENT;for(i=0;i<n;i++)s+=x[i]*x[i];s=sqrtf(s/(float)n+e);for(i=0;i<n;i++)x[i]/=s;return VAIST_OK;}
VAIST_API VaistStatus vaist_nn_softmax_f32(float*x,size_t n){size_t i;float m,s=0;if(!ok(x,n))return VAIST_INVALID_ARGUMENT;m=x[0];for(i=1;i<n;i++)if(x[i]>m)m=x[i];for(i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}for(i=0;i<n;i++)x[i]/=s;return VAIST_OK;}
VAIST_API VaistStatus vaist_nn_rope_f32(float*x,size_t pairs,float theta,size_t pos){size_t i;if(!x||!pairs||theta<=0)return VAIST_INVALID_ARGUMENT;for(i=0;i<pairs;i++){float inv=powf(theta,-(2.0f*(float)i/(float)(pairs*2u)));float a=(float)pos*inv,c=cosf(a),s=sinf(a),u=x[2*i],v=x[2*i+1];x[2*i]=u*c-v*s;x[2*i+1]=u*s+v*c;}return VAIST_OK;}
