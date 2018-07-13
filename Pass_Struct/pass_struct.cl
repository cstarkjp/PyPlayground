typedef struct VglClStrEl{
    float data[VGL_ARR_CLSTREL_SIZE];
    int ndim;
    int shape[VGL_ARR_SHAPE_SIZE];
    int offset[VGL_ARR_SHAPE_SIZE];
    int size;
} VglClStrEl;

typedef struct VglClShape{
    int ndim;
    int shape[VGL_ARR_SHAPE_SIZE];
    int offset[VGL_ARR_SHAPE_SIZE];
    int size;
} VglClShape;

__kernel void pass_struct(
        __global uchar *img_input,
        __global uchar *img_output,
        __constant VglClShape* img_shape,
        __constant VglClStrEl* window
        )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
//    __constant VglClShape *vgclshape=img_shape;

    if (global_id==0) {
        printf("In GPU (running):\n Kernel instance = %d\n", global_id);
//        printf(" VglClShape @ %p = %x bytes\n", img_shape, sizeof(*img_shape) );
        printf(" VglClShape: %d, %d, %d, %d, %d, %d, %d, %d\n",
            img_shape->ndim,
            img_shape->shape[0], img_shape->shape[1], img_shape->shape[2],
            img_shape->offset[0], img_shape->offset[1], img_shape->offset[2],
            img_shape->size);
    }
    return;
}

__kernel void get_struct_sizes( __global uint *struct_sizes )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    VglClStrEl vgclstrel;
    VglClShape vgclshape;
    uint offset;

    printf("In GPU (probing):\n Kernel instance = %d\n", global_id);

    if (global_id==0) {
        offset = (uint)&(vgclstrel.data);
        struct_sizes[0] = (uint)sizeof(vgclstrel);
        struct_sizes[1] = (uint)&(vgclstrel.ndim)-offset;
        struct_sizes[2] = (uint)&(vgclstrel.shape)-offset;
        struct_sizes[3] = (uint)&(vgclstrel.offset)-offset;
        struct_sizes[4] = (uint)&(vgclstrel.size)-offset;
        offset = (uint)&(vgclshape.ndim);
        struct_sizes[5] = (uint)sizeof(vgclshape);
        struct_sizes[6] = (uint)&(vgclshape.shape)-offset;
        struct_sizes[7] = (uint)&(vgclshape.offset)-offset;
        struct_sizes[8] = (uint)&(vgclshape.size)-offset;
    }
    return;
}

//printf(" VglClStrEl:  size %d  %d, %d, %d, %d\n",
//    struct_sizes[0],
//    struct_sizes[1],
//    struct_sizes[2],
//    struct_sizes[3],
//    struct_sizes[4]);
//}
//printf(" VglClShape:  size=%d  %d, %d, %d\n",
//    struct_sizes[5],
//    struct_sizes[6],
//    struct_sizes[7],
//    struct_sizes[8]);

