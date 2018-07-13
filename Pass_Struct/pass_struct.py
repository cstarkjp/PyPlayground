#!/usr/bin/env python3

"""
Demo passing of struct from Python to OpenCL kernel
"""

from argparse import ArgumentParser,ArgumentTypeError,ArgumentDefaultsHelpFormatter
import pyopencl as cl
# import pyopencl.array
import pyopencl.tools as cltools
import numpy as np
import warnings
import os
os.environ['PYOPENCL_COMPILER_OUTPUT']='0'

pdebug = print

COPY_READ_ONLY  = cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR
COPY_READ_WRITE = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
COPY_WRITE      = cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR

def run():
    cl_platform = 0
    cl_device = 2
    cl_state = Initialize_CL(cl_platform,cl_device)  
    cl_state.n_workitems_per_workgroup = 32
  
    vgclstrel_sizes, vgclshape_sizes = get_struct_sizes(cl_state)
    use_struct(cl_state, vgclshape_sizes)

def get_struct_sizes(cl_state):
    cl_state.kernel_fn = 'get_struct_sizes'
    cl_files = ['pass_struct.cl']
    cl_src_path = '.'
    cl_state.kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_state.kernel_source += fp.read()

    # Define arrays and implicitly define n_kernel_instances
    vgl_arr_clstrel_size = cl_state.n_workitems_per_workgroup*1
    vgl_arr_shape_size = 20
    compile_options_dict = {
        'VGL_ARR_CLSTREL_SIZE' : (vgl_arr_clstrel_size,'u'),
        'VGL_ARR_SHAPE_SIZE'   : (vgl_arr_shape_size,'u')
    }
    global_size = [1,1]
    local_size  = [1,1]
    
    # Compile CL
    # Specify macros to  be passed to compiler
    cl_state.compile_options = set_compile_options(compile_options_dict)
    # Build
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl_state.program = cl.Program(cl_state.context, cl_state.kernel_source)\
                             .build(options=cl_state.compile_options)
     # Report compiler warnings, errors
    report_build_log(cl_state.program, cl_state.device)
    
    # Set the GPU kernel
    cl_state.kernel = getattr(cl_state.program,cl_state.kernel_fn)
    
    # Buffers
    struct_sizes  = np.zeros(9, dtype=np.uint32)
    struct_sizes_buffer = cl.Buffer(cl_state.context, COPY_WRITE, 
                                        hostbuf=struct_sizes)
#     dummy = np.array((1),dtype=np.uint32)
#     dummy_buffer = cl.Buffer(cl_state.context, COPY_READ_ONLY, hostbuf=dummy)
    
    # Pass buffers to GPU
    buffer_list = [struct_sizes_buffer]
    cl_state.kernel.set_args(*buffer_list)
    cl_state.kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    
#     report_kernel_info(cl_state)
#     report_device_info(cl_state)
    
    # Do the GPU compute
    cl_state.event = cl.enqueue_nd_range_kernel(cl_state.queue, cl_state.kernel, 
                                       global_size, local_size)
    
    # Fetch the data back from the GPU and finish
    cl.enqueue_copy(cl_state.queue, struct_sizes, struct_sizes_buffer)
    cl_state.queue.finish()
    vgclstrel_sizes = struct_sizes[:5]
    vgclshape_sizes = struct_sizes[5:]
    print(' VglClStrEl size={0} data=0 ndim=+{1} shape=+{2} offset=+{3} size=+{4}\n'
          .format( *[addr for addr in vgclstrel_sizes] ))
    print(' VglClShape size={0} ndim=0 shape=+{1} offset=+{2} size=+{3}\n'
          .format( *[addr for addr in vgclshape_sizes] ))
    return vgclstrel_sizes, vgclshape_sizes

def use_struct(cl_state, vgclshape_sizes):
    cl_state.kernel_fn = 'pass_struct'
    cl_files = ['pass_struct.cl']
    cl_src_path = '.'
    cl_state.kernel_source = ''
    for cl_file in cl_files:
        with open(os.path.join(cl_src_path,cl_file), 'r') as fp:
            cl_state.kernel_source += fp.read()

    # Define arrays and implicitly define n_kernel_instances
    n_workitems_per_workgroup = 32
    vgl_arr_clstrel_size = n_workitems_per_workgroup*4
    vgl_arr_shape_size = 20
    compile_options_dict = {
        'VGL_ARR_CLSTREL_SIZE' : (vgl_arr_clstrel_size,'u'),
        'VGL_ARR_SHAPE_SIZE'   : (vgl_arr_shape_size,'u')
    }
    global_size = [vgl_arr_clstrel_size,1]
    local_size  = [n_workitems_per_workgroup,1]
    
    # Compile CL
    # Specify macros to  be passed to compiler
    cl_state.compile_options = set_compile_options(compile_options_dict)
    # Build
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cl_state.program = cl.Program(cl_state.context, cl_state.kernel_source)\
                             .build(options=cl_state.compile_options)
     # Report compiler warnings, errors
    report_build_log(cl_state.program, cl_state.device)
    
    # Set the GPU kernel
    cl_state.kernel = getattr(cl_state.program,cl_state.kernel_fn)
            
    # Buffers
    img_input  = np.zeros(vgl_arr_clstrel_size, dtype=np.uint8)
    img_output = np.zeros(vgl_arr_clstrel_size, dtype=np.uint8)
    img_shape  = np.zeros((vgclshape_sizes[0]), dtype=np.uint8)
    window     = np.zeros((vgclshape_sizes[0]), dtype=np.uint8)
    
    # Set some values
    pdebug('vgclshape_sizes',vgclshape_sizes)
    ndim   = np.array([3], dtype=np.int32)
    shape    = np.zeros((vgl_arr_shape_size,), dtype=np.int32)
    shape[:] = np.random.random_integers(1, high=2**32-1, size=vgl_arr_shape_size)
    offset    = np.zeros((vgl_arr_shape_size,), dtype=np.int32)
    offset[:] = np.random.random_integers(1, high=2**32-1, size=vgl_arr_shape_size)
    size   = np.array([vgl_arr_clstrel_size], dtype=np.int32)
#     vgclshape_struct = np.dtype([('ndim', np.int32), 
#                                  ('size', np.int32)])
#     vgclshape = np.empty(1, vgclshape_struct)
#     vgclshape['ndim'] = 2
#     vgclshape['size'] = 196
#     img_shape.setfield(vgclshape,vgclshape_struct,0)
    def copy_into_byte_array(value, byte_array, offset):
        for i,b in enumerate(np.ndarray.tobytes(value)):
            byte_array[i+offset] = b
    copy_into_byte_array(ndim,   img_shape, 0) 
    copy_into_byte_array(shape,  img_shape, vgclshape_sizes[1]) 
    copy_into_byte_array(offset, img_shape, vgclshape_sizes[2]) 
    copy_into_byte_array(size,   img_shape, vgclshape_sizes[3]) 
#     pdebug(img_shape)
    pdebug(ndim,
           shape[0], shape[1], shape[2], 
           offset[0], offset[1], offset[2],
           size)
    
    
    # Pass buffers to GPU
    img_input_buffer  = cl.Buffer(cl_state.context, COPY_READ_WRITE,hostbuf=img_input)
    img_output_buffer = cl.Buffer(cl_state.context, COPY_READ_WRITE,hostbuf=img_output)
    img_shape_buffer  = cl.Buffer(cl_state.context, COPY_READ_ONLY, hostbuf=img_shape)
    window_buffer     = cl.Buffer(cl_state.context, COPY_READ_ONLY, hostbuf=window)
    buffer_list = [img_input_buffer, img_output_buffer, img_shape_buffer, window_buffer]
    cl_state.kernel.set_args(*buffer_list)
    cl_state.kernel.set_scalar_arg_dtypes( [None]*len(buffer_list) )
    
#     report_kernel_info(cl_state)
#     report_device_info(cl_state)
    
    # Do the GPU compute
    cl_state.event = cl.enqueue_nd_range_kernel(cl_state.queue, cl_state.kernel, 
                                       global_size, local_size)
    
    # Fetch the data back from the GPU and finish
#     cl.enqueue_copy(cl_state.queue, img_output, img_output_buffer)
    cl_state.queue.finish()   

def make_cl_dtype(cl_state,name,dtype):
    """
    Generate an OpenCL structure typedef codelet from a numpy structured 
    array dtype.
    
    Args:
        cl_state (obj):
        name (str):
        dtype (numpy.dtype):
    
    Returns:
        numpy.dtype, pyopencl.dtype, str: 
            processed dtype, cl dtype, CL typedef codelet
    """
    processed_dtype, c_decl \
        = cltools.match_dtype_to_c_struct(cl_state.device, name, dtype)
    return processed_dtype, cltools.get_or_register_dtype(name, processed_dtype), c_decl

class Initialize_CL():
    def __init__(self, which_cl_platform, which_cl_device ):
        # Prepare CL essentials
        self.cl_platform = which_cl_platform
        self.cl_device = which_cl_device
        self.platform, self.device, self.context \
            = prepare_cl_context(which_cl_platform,which_cl_device)
        self.queue = cl.CommandQueue(self.context,
                                properties=cl.command_queue_properties.PROFILING_ENABLE)

def prepare_cl_context(cl_platform=0, cl_device=2):
    """
    Prepare PyOpenCL platform, device and context.
    
    Args:
        cl_platform (int):
        cl_device (int):
    
    Returns:
        pyopencl.Platform, pyopencl.Device, pyopencl.Context:
            PyOpenCL platform, PyOpenCL device, PyOpenCL context
    """
    cl_platform, cl_device = choose_platform_and_device(cl_platform,cl_device)
    platform = cl.get_platforms()[cl_platform]
    devices = platform.get_devices()
    device = devices[cl_device]
    context = cl.Context([device])
    return platform, device, context

def choose_platform_and_device(cl_platform='env',cl_device='env'):
    """
    Get OpenCL platform & device from environment variables if they are set.
    
    Args:
        cl_platform (int):
        cl_device (int):
    
    Returns:
        int, int:
            CL platform, CL device
    """
    if cl_platform=='env':
        try:
            cl_platform = int(environ['PYOPENCL_CTX'].split(':')[0])
        except:
            cl_platform = 0
    if cl_device=='env':
        try:
            cl_device = int(environ['PYOPENCL_CTX'].split(':')[1])
        except:
            cl_device = 2
    return cl_platform, cl_device

def prepare_cl_queue(context=None, cl_kernel_source=None, compile_options=[]):
    """
    Build PyOpenCL program and prepare command queue.
    
    Args:
        context (pyopencl.Context): GPU/OpenCL device context
        cl_kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Program, pyopencl.CommandQueue: 
            PyOpenCL program, PyOpenCL command queue
    """
#     compile_options = ['-cl-fast-relaxed-math',
#                        '-cl-single-precision-constant',
#                        '-cl-unsafe-math-optimizations',
#                        '-cl-no-signed-zeros',
#                        '-cl-finite-math-only']
    program = cl.Program(context, cl_kernel_source).build(cache_dir=False,
                                                          options=compile_options)
    queue = cl.CommandQueue(context)
    return program, queue

def prepare_cl(cl_platform=0, cl_device=2, cl_kernel_source=None, compile_options=[]):
    """
    Prepare PyOpenCL stuff.
    
    Args:
        cl_device (int):
        cl_kernel_source (str): OpenCL kernel code string
    
    Returns:
        pyopencl.Platform, pyopencl.Device, pyopencl.Context, pyopencl.Program, \
        pyopencl.CommandQueue:
            PyOpenCL platform, PyOpenCL device, PyOpenCL context, PyOpenCL program, \
            PyOpenCL command queue
    """
    platform,device,context = prepare_cl_context(cl_platform,cl_device)
    program,queue = prepare_cl_queue(context,cl_kernel_source,compile_options)
    return platform, device, context, program, queue

def set_compile_options(compile_options_dict):
    """
    Convert the info struct into a list of '-D' compiler macros.
    
    Args:
        options_dict (dict): 
        
    Returns:
        list:
            compile options
    """
    rtn_list = []
    for item in compile_options_dict.items():
        name  = item[0].upper()
        value = item[1][0]
        type  = '' #item[1][1]
        list_item = ['-D','{0}={1}{2}'.format(name,value,type)]
        rtn_list += list_item
    pdebug(rtn_list)
    return rtn_list

def report_kernel_info(cl_state):
    """
    Fetch and print GPU/OpenCL kernel info.
    
    Args:
        cl_state (obj):
    """
    device = cl_state.device
    kernel = cl_state.kernel
    # Report some GPU info
    print('Kernel reference count:',
          kernel.get_info(
              cl.kernel_info.REFERENCE_COUNT))
    print('Kernel number of args:',
          kernel.get_info(
              cl.kernel_info.NUM_ARGS))
    print('Kernel function name:',
          kernel.get_info(
              cl.kernel_info.FUNCTION_NAME))
    print('Maximum work group size:',
          kernel.get_work_group_info(
              cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
    print('Recommended work group size:',
          kernel.get_work_group_info(
              cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device))
    print('Local memory size:',
          kernel.get_work_group_info(
              cl.kernel_work_group_info.LOCAL_MEM_SIZE, device), 'bytes')
    print('Private memory size:',
          kernel.get_work_group_info(
              cl.kernel_work_group_info.PRIVATE_MEM_SIZE, device), 'bytes')    

def report_device_info(cl_state):
    """
    Fetch and print GPU/OpenCL device info.
    
    Args:
        cl_platform (int):
        cl_device (int):
        platform (pyopencl.Platform):
        device (pyopencl.Device):
    """
    cl_platform = cl_state.cl_platform
    cl_device   = cl_state.cl_device
    platform = cl_state.platform    
    device   = cl_state.device
    print('OpenCL platform #{0} = {1}'.format(cl_platform,platform))
    print('OpenCL device #{0} = {1}'.format(cl_device,device))
    n_bytes = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
    print('Global memory size: {} bytes = {}'.format(n_bytes,neatly(n_bytes)))
    n_bytes = device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    print('Max memory alloc size: {} bytes = {}'
                        .format(n_bytes,neatly(n_bytes)))
    
    device_info_list = [s for s in dir(cl.device_info) if not s.startswith('__')]
    for s in device_info_list:
        try:
            print('{0} = {1}'.format(s,device.get_info(getattr(cl.device_info,s))))
        except:
            pass

def report_build_log(program, device):
    """
    Fetch and print GPU/OpenCL program build log.
    
    Args:
        program (pyopencl.Program):
        device (pyopencl.Device):
    """
    build_log = program.get_build_info(device,cl.program_build_info.LOG)
    if len(build_log.replace(' ',''))>0:
        print('\nOpenCL build log: {}'.format(build_log))            
    
def neatly(byte_size):
    """Returns a human readable string reprentation of bytes"""
    units=['B ','kB','MB','GB']  # Note: actually MiB etc
    for unit in units:
        if byte_size>=1024:
            byte_size = byte_size/1024.0
        else:
            break
    if unit=='GB':
        return str(int(0.5+10*byte_size)/10)+unit
    else:
        return str(int(0.5+byte_size))+unit

def _parse_cmd_line_args():
    """
    Parse the command line arguments using :mod:`argparse`.
    The arguments are assumed to be passed via `_sys.argv[1:]`.

    Return:
        :obj:`argparse.Namespace`:  parsed command line arguments
    """
    usage = '''Demo'''
    parser = ArgumentParser(description=usage, 
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-a', '--analysis', dest='do_analysis',
                        default=None, type=_str2bool,  action="store", 
                        metavar='analysis_flag',
                        help='analyze streamline patterns, distributions')
    
    parser.add_argument('-f', '--file', dest='parameters_file',
                        default=None, type=str,  action="store",  
                        metavar='parameters_file',
                        help='import JSON parameters file')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
#     kwargs = vars(_parse_cmd_line_args())
#     run(**kwargs)
    run()
