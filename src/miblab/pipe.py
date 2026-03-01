# Standard Library
import argparse
import logging
import os
import re
import sys
from pathlib import Path
import gc

# Third-Party Packages
import dask
import psutil
from dask.distributed import Client, LocalCluster
from joblib import parallel_backend



def get_dask_client(min_ram_per_worker = 4.0): # Increase this for heavier data
    # We want at least 4GB per worker. If we don't have enough RAM 
    # for all CPUs, we reduce the number of workers.
    """
    Synchronizes Dask with SLURM allocation.

    This will reduce the number of workers to satisfy the minimal 
    RAM requirement set. If this is not possible and error is raised.

    This function is necessary on HPC because it will let 
    dask know that the total RAM is to be distributed 
    over all workers. Otherwise dask may assume the total RAM is 
    available to each worker on the HPC, and cause memory overflow.
    """
    # Importing here because not all pipelines have dask/psutil installed

    # 1. Detect CPUs (Slurm-aware)
    # SLURM_CPUS_PER_TASK is set by --cpus-per-task
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    
    # 2. Detect RAM (Slurm-aware)
    # SLURM_MEM_PER_NODE is set by --mem
    slurm_mem_str = os.environ.get("SLURM_MEM_PER_NODE")
    
    if slurm_mem_str:
        # Parse Slurm memory string (e.g., '64G', '128000M', '64000')
        mem_digits = int(re.search(r'\d+', slurm_mem_str).group())
        if 'G' in slurm_mem_str.upper():
            total_ram_gb = mem_digits
        elif 'M' in slurm_mem_str.upper():
            total_ram_gb = mem_digits / 1024
        else:
            # Slurm default is usually MB if no unit specified
            total_ram_gb = mem_digits / 1024
    else:
        # Laptop fallback: Use total physical RAM
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

    # 3. Calculate per-worker limit and BALANCE cores vs RAM
    # Calculate how many workers we can actually afford
    # Leave 40% for overheads
    max_affordable_workers = int((total_ram_gb * 0.6) // min_ram_per_worker)
    
    # n_workers becomes the lower of 'what we have' vs 'what we can afford'
    if min(n_workers, max_affordable_workers) < 1:
        raise RuntimeError(
            f"There is not enough memory available to allocate at least"
            f"{min_ram_per_worker} GB per worker. You have a maximum of {n_workers} workers "
            f"but only {total_ram_gb} GB memory in total. You need to either "
            f"reduce the minimal RAM per worker or increase the total RAM."
        )
    n_workers = min(n_workers, max_affordable_workers)
    memory_limit_per_worker = f"{total_ram_gb / n_workers:.2f}GB"
    
    logging.info(f"Dask Sync: {n_workers} workers | {total_ram_gb:.1f}GB total | {memory_limit_per_worker}/worker")

    # 4. Silence internal library threading (Very Important!)
    # This prevents NumPy/MKL from spawning 8 threads per Dask worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1, # 1 thread per worker is best for heavy NumPy/Math
        memory_limit=memory_limit_per_worker
    )
    return Client(cluster)




def adjust_workers(client, min_ram_per_worker=4.0, overhead_ram=8):
    """
    Re-calculates and scales workers while respecting SLURM/Hardware ceilings.
    Ensures we stay within the 'affordable' memory range.
    """

    
    # 1. Re-detect the Ceiling (What we have available)
    # Respect SLURM allocation first, fall back to physical cores
    max_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))
    
    # 2. Re-detect Total RAM (Slurm-aware)
    slurm_mem_str = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem_str:
        mem_digits = int(re.search(r'\d+', slurm_mem_str).group())
        total_ram_gb = mem_digits if 'G' in slurm_mem_str.upper() else mem_digits / 1024
    else:
       
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

    if overhead_ram >= total_ram_gb:
        raise ValueError(
            f"The overhead RAM of {overhead_ram} GB is more than the total available RAM of {total_ram_gb} GB.\n"
            f"Solutions: reduce overhead_ram, or (if working on HPC) ask the scheduler for more RAM."
        )

    total_ram_gb -= overhead_ram  

    if min_ram_per_worker <= total_ram_gb:
        raise ValueError(
            f"Insufficient memory for a single worker.\n"
            f"Solutions: reduce overhead_ram, reduce min_ram_per_worker, or (if working on HPC) ask the scheduler for more RAM."
        )

    # 3. Calculate 'Affordable' workers based on new RAM requirements
    # Using 80% of total RAM as the budget, similar to your original setup
    max_affordable_workers = int(total_ram_gb // min_ram_per_worker)
    
    # 4. Final Worker Count: The lower of 'Cores we have' vs 'RAM we can afford'
    n_workers = min(max_cores, max_affordable_workers)
    
    if n_workers < 1:
        logging.error(f"Cannot satisfy {min_ram_per_worker}GB requirement with {total_ram_gb:.1f}GB total RAM.")
        return # Or raise RuntimeError to stop the pipeline

    # 5. Calculate new limit string for Dask
    # Using 90% as per your original logic to give the OS some breathing room
    memory_limit_per_worker = f"{total_ram_gb / n_workers:.2f}GB"

    # 6. Apply Scaling
    logging.info(f"Adjusting: {n_workers} workers at {memory_limit_per_worker} each.")
    
    # Note: LocalCluster allows scaling n_workers. 
    # To change memory_limit per worker dynamically, we update the cluster property.
    client.cluster.scale(n_workers)
    
    # Wait for the cluster to stabilize
    client.wait_for_workers(n_workers, timeout=30)


# def setup_logging(build, pipeline):
#     dir_logs = os.path.join(build, pipeline)
#     os.makedirs(dir_logs, exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(f"{dir_logs}.log"),
#             logging.StreamHandler(sys.stdout),
#         ],
#     )

    

def stage_output_dir(build, pipeline, module):

    # Outputs of the stage
    stage = Path(module).name[:-3]
    dir_output = os.path.join(build, pipeline, stage)
    os.makedirs(dir_output, exist_ok=True)
    return dir_output


def ppln_output_dir(build, pipeline):

    # Outputs of the ppln
    dir_output = os.path.join(build, pipeline)
    os.makedirs(dir_output, exist_ok=True)
    return dir_output


# def run_dask_script(run, default_build, pipeline, min_ram_per_worker = 4.0):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--build", type=str, default=default_build, help="Build folder")
#     args = parser.parse_args()

#     setup_logging(args.build, pipeline)

#     client = get_dask_client(min_ram_per_worker = min_ram_per_worker)
#     run(args.build, client)
#     client.close()


# def run_dask_stage(run, default_build, pipeline, module, min_ram_per_worker = 4.0, **args):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--build", type=str, default=default_build, help="Build folder")
#     for arg, kwargs in args.items():
#         parser.add_argument(f"--{arg}", **kwargs)
#     args = parser.parse_args()

#     dir_output = stage_output_dir(args.build, pipeline, module)
#     logfile = os.path.join(dir_output, 'log.log')

#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(logfile),
#             logging.StreamHandler(sys.stdout),
#         ],
#     )

#     client = get_dask_client(min_ram_per_worker = min_ram_per_worker)
#     optional_args = {k: v for k, v in vars(args).items() if k != 'build'}

#     try:
#         with parallel_backend('dask'):
#             run(args.build, client, **optional_args)
#     finally:
#         # 1. Wait for joblib/dask to settle
#         logging.info("Shutting down Dask client and cluster...")
        
#         # 2. Get the cluster reference before closing the client
#         cluster = getattr(client, 'cluster', None)
        
#         # 3. Close the client first with a timeout
#         # This tells workers to stop without immediately nuking the network
#         client.close(timeout=60) 
        
#         # 4. Explicitly scale the cluster to zero before closing
#         # This is often more successful on Slurm/Stanage
#         if cluster is not None:
#             if hasattr(cluster, 'scale'):
#                 cluster.scale(0)
#             cluster.close(timeout=60)

#         # 5. Clean up local memory
#         gc.collect()
#         logging.info("Shutdown complete.")


# def run_script(run, default_build, pipeline):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--build", type=str, default=default_build, help="Build folder")
#     args = parser.parse_args()

#     setup_logging(args.build, pipeline)

#     run(args.build)






def run_client_ppln(run, default_build, pipeline, min_ram_per_worker = 4.0, **args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=default_build, help="Build folder")
    for arg, kwargs in args.items():
        parser.add_argument(f"--{arg}", **kwargs)
    args = parser.parse_args()

    dir_output = ppln_output_dir(args.build, pipeline)
    logfile = os.path.join(dir_output, 'log.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    client = get_dask_client(min_ram_per_worker = min_ram_per_worker)
    optional_args = {k: v for k, v in vars(args).items() if k != 'build'}

    try:
        with parallel_backend('dask'):
            run(args.build, logfile, client, **optional_args)
    finally:
        # 1. Wait for joblib/dask to settle
        logging.info("Shutting down Dask client and cluster...")
        
        # 2. Get the cluster reference before closing the client
        cluster = getattr(client, 'cluster', None)
        
        # 3. Close the client first with a timeout
        # This tells workers to stop without immediately nuking the network
        client.close(timeout=60) 
        
        # 4. Explicitly scale the cluster to zero before closing
        # This is often more successful on Slurm/Stanage
        if cluster is not None:
            if hasattr(cluster, 'scale'):
                cluster.scale(0)
            cluster.close(timeout=60)

        # 5. Clean up local memory
        gc.collect()
        logging.info("Shutdown complete.")


def run_ppln(run, default_build, pipeline, **args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=default_build, help="Build folder")
    for arg, kwargs in args.items():
        parser.add_argument(f"--{arg}", **kwargs)
    args = parser.parse_args()

    dir_output = ppln_output_dir(args.build, pipeline)
    logfile = os.path.join(dir_output, 'log.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    optional_args = {k: v for k, v in vars(args).items() if k != 'build'}
    run(args.build, **optional_args)


def run_client_stage(run, default_build, pipeline, module, min_ram_per_worker = 4.0, **args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=default_build, help="Build folder")
    for arg, kwargs in args.items():
        parser.add_argument(f"--{arg}", **kwargs)
    args = parser.parse_args()

    dir_output = stage_output_dir(args.build, pipeline, module)
    logfile = os.path.join(dir_output, 'log.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    client = get_dask_client(min_ram_per_worker = min_ram_per_worker)
    optional_args = {k: v for k, v in vars(args).items() if k != 'build'}

    try:
        with parallel_backend('dask'):
            run(args.build, logfile, client, **optional_args)
    finally:
        # 1. Wait for joblib/dask to settle
        logging.info("Shutting down Dask client and cluster...")
        
        # 2. Get the cluster reference before closing the client
        cluster = getattr(client, 'cluster', None)
        
        # 3. Close the client first with a timeout
        # This tells workers to stop without immediately nuking the network
        client.close(timeout=60) 
        
        # 4. Explicitly scale the cluster to zero before closing
        # This is often more successful on Slurm/Stanage
        if cluster is not None:
            if hasattr(cluster, 'scale'):
                cluster.scale(0)
            cluster.close(timeout=60)

        # 5. Clean up local memory
        gc.collect()
        logging.info("Shutdown complete.")


def run_stage(run, default_build, pipeline, module, **args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=default_build, help="Build folder")
    for arg, kwargs in args.items():
        parser.add_argument(f"--{arg}", **kwargs)
    args = parser.parse_args()

    dir_output = stage_output_dir(args.build, pipeline, module)
    logfile = os.path.join(dir_output, 'log.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    optional_args = {k: v for k, v in vars(args).items() if k != 'build'}
    run(args.build, logfile, **optional_args)

