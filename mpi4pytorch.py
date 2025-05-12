"""                                              mpi4pytorch.py
 This module contains convenience methods that make it easy to use mpi4py.  The available functions handle memory
 allocation and other data formatting tasks so that tensors can be easily reduced/broadcast using 1 line of code.
"""

import numpy as np
import mpi4py # Ensure mpi4py is imported if it's going to be used directly.
# from mpi4py import MPI # This is often how MPI is imported.

def setup_MPI():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        #  Convert the Object to a Class so that it is possible to add attributes later
        #  This class A modification might not be standard or necessary for basic MPI ops.
        #  If it causes issues, consider using comm directly.
        class A(mpi4py.MPI.Intracomm):
            pass
        comm = A(comm)
    except ImportError:
       print("mpi4py not found, MPI operations will be no-ops.")
       comm = None
    except Exception as e:
       print(f"Error setting up MPI: {e}")
       comm = None
    return comm


def print_once(comm, *message):
    if not comm or comm.Get_rank()==0:
        print (''.join(str(i) for i in message))

def is_master(comm):
    return not comm or comm.Get_rank()==0

def allreduce_max(comm, array, display_info=False):
    if not comm:
        return array
    array_np = np.asarray(array, dtype='d') # 'd' corresponds to float64
    total = np.zeros_like(array_np)
    # FIX: Replace np.float with np.float64 or float
    float_min = np.finfo(np.float64).min 
    total.fill(float_min)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array_np), array_np.nbytes))
        rows = str(comm.gather(array_np.shape[0]))
        cols_str = ""
        if array_np.ndim > 1:
            cols_str = str(comm.gather(array_np.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols_str))

    comm.Allreduce(array_np, total, op=mpi4py.MPI.MAX)
    return total

def allreduce_min(comm, array, display_info=False):
    if not comm:
        return array
    array_np = np.asarray(array, dtype='d')
    total = np.zeros_like(array_np)
    # FIX: Replace np.float with np.float64 or float
    float_max = np.finfo(np.float64).max
    total.fill(float_max)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array_np), array_np.nbytes))
        rows = str(comm.gather(array_np.shape[0]))
        cols_str = ""
        if array_np.ndim > 1:
            cols_str = str(comm.gather(array_np.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols_str))


    comm.Allreduce(array_np, total, op=mpi4py.MPI.MIN)
    return total


def reduce_max(comm, array, display_info=False):
    if not comm:
        return array
    array_np = np.asarray(array, dtype='d') # Ensure it's a numpy array for MPI
    total = np.zeros_like(array_np)
    # FIX: Replace np.float with np.float64 or float
    float_min = np.finfo(np.float64).min
    total.fill(float_min)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array_np), array_np.nbytes))
        rows = str(comm.gather(array_np.shape[0]))
        cols_str = ""
        if array_np.ndim > 1:
            cols_str = str(comm.gather(array_np.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols_str))

    # Ensure using the numpy buffer for MPI
    comm.Reduce(array_np, total, op=mpi4py.MPI.MAX, root=0)
    return total

def reduce_min(comm, array, display_info=False):
    if not comm:
        return array
    array_np = np.asarray(array, dtype='d')
    total = np.zeros_like(array_np)
    # FIX: Replace np.float with np.float64 or float
    float_max = np.finfo(np.float64).max
    total.fill(float_max)

    if display_info:
        print ("(%d): sum=%f : size=%d"%(get_rank(comm), np.sum(array_np), array_np.nbytes))
        rows = str(comm.gather(array_np.shape[0]))
        cols_str = ""
        if array_np.ndim > 1:
            cols_str = str(comm.gather(array_np.shape[1]))
        print_once(comm, "reduce: %s, %s"%(rows, cols_str))

    comm.Reduce(array_np, total, op=mpi4py.MPI.MIN, root=0)
    return total

def barrier(comm):
    if not comm:
        return
    try:
        comm.barrier() # Or comm.Barrier() depending on mpi4py version/style
    except AttributeError:
        comm.Barrier()


def get_mpi_info():
    try:
        vendor, version = mpi4py.MPI.get_vendor()
        return f"{vendor} {version}"
    except ImportError:
        return "mpi4py not installed"
    except Exception as e:
        return f"Error getting MPI info: {e}"


def get_rank(comm):
    try:
        return comm.Get_rank()
    except ImportError:
        return 0 # Default for non-MPI runs
    except Exception:
        return 0


def get_num_procs(comm):
    try:
        return comm.Get_size()
    except ImportError:
        return 1 # Default for non-MPI runs
    except Exception:
        return 1
