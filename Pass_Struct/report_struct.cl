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

__kernel void report_struct(
        __global unsigned char *img_input,
        __global unsigned char *img_output
        )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    VglClStrEl vgclstrel;
    VglClShape vgclshape;

    if (global_id==0) {
        printf("In GPU:\n Kernel instance = %d\n",
            global_id);
        printf(" VglClShape @ %p = %x bytes\n",
            &vgclstrel, sizeof(vgclstrel) );
        printf(" VglClShape @ %p: %p, %p, %p, %p\n",
            &vgclstrel, &(vgclstrel.ndim), &(vgclstrel.shape),
            &(vgclstrel.offset), &(vgclstrel.size));
    }
    return;
}

