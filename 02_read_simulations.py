import os

import numpy

from simulation_settings import parameter_range, simulation_types, \
    n_repetitions, n_generations, n_flights_per_generation, \
    p_goal_range, p_social_range, p_memory_range, p_continuity_range_used, \
    sd_goal_range, sd_continuity, sd_social, \
    sd_memory_max_range, sd_memory_min, sd_memory_steps, \
    start_pos, goal_pos, n_landmarks, start_heading, \
    stepsize, max_steps, goal_threshold, landmark_threshold

# Set this to True to overwrite the existing .dat files.
OVERWRITE_EXISTING_TMP = False

# Set the maximum path lengths.
MAX_PATH_LENGTH = 200

# Compute the expected number of files in each run's directory.
EXPECTED_N_SUB_DIR_FILES = n_generations + \
    n_generations*n_flights_per_generation
    
# Files and folders.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data_simulation_{}".format(parameter_range))
OUTDIR = os.path.join(DIR, "output_simulation_{}".format(parameter_range))
TMP_DIR = os.path.join(OUTDIR, "reduced_data")
for outpath in [OUTDIR, TMP_DIR]:
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
LOAD_ERROR_LOG = os.path.join(OUTDIR, "data_loading_errors.csv")

# Failsafe.
if not OVERWRITE_EXISTING_TMP:
    raise Exception("Failsafe: set 'OVERWRITE_EXISTING_TMP' to True to " + \
        "load simulation data. This will overwrite existing .dat files.")

# Write a header to the error log, which also clears pre-existing data in that
# file.
with open(LOAD_ERROR_LOG, "w") as f:
    line = ["simulation", "run", "n_files"]
    f.write(",".join(line))


# # # # #
# LOADING FUNCTIONS

def buf_count_newlines_gen(fpath):
    # Fastest counter from: https://stackoverflow.com/questions/845058/
    # how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b
    with open(fpath, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count

def load_efficiency_csv(fpath):
    # Read number of newlines in file, which should be exactly the number of
    # data lines (the final line does not end on a newline).
    n_lines = buf_count_newlines_gen(fpath)
    # Create NumPy arrays to hold the data.
    efficiency = numpy.zeros((2,n_lines), dtype=numpy.float64)
    # Load data from file.
    with open(fpath, "r") as f:
        # Get the header out first.
        header = f.readline()
        # Now read all data.
        for i, line in enumerate(f):
            # Parse the line.
            line = line.rstrip("\n").split(",")
            efficiency[0,i] = float(line[1])
            efficiency[1,i] = float(line[2])
    
    return efficiency

def load_xy_csv(fpath):
    # Read number of newlines in file, which should be exactly the number of
    # data lines (the final line does not end on a newline).
    n_lines = buf_count_newlines_gen(fpath)
    # Create NumPy arrays to hold the data.
    x = numpy.zeros((2,n_lines), dtype=numpy.float64)
    y = numpy.zeros((2,n_lines), dtype=numpy.float64)
    # Load data from file.
    with open(fpath, "r") as f:
        # Get the header out first.
        header = f.readline()
        # Now read all data.
        for i, line in enumerate(f):
            # Parse the line.
            line = line.rstrip("\n").split(",")
            x[0,i] = float(line[0])
            y[0,i] = float(line[1])
            x[1,i] = float(line[2])
            y[1,i] = float(line[3])
    
    return x, y


# # # # #
# LOAD DATA

# Get the list of all folders.
all_data_directories = os.listdir(DATADIR)
not_data_directories = []
# Parse folder names, which should look like this:
# "Pgoal-10_SDgoal-10_Pcontinuity-10_SDcontinuity-15_Psocial-10_SDsocial-40_
# Pmemory-70_SDmemoryMax-100_SDmemoryMin-50_SDmemorySteps-5"
all_parameters = { \
    "Pgoal": [], \
    "Pcontinuity": [], \
    "Psocial": [], \
    "Pmemory": [], \
    "SDgoal": [], \
    "SDcontinuity": [], \
    "SDsocial": [], \
    "SDmemoryMax": [], \
    "SDmemoryMin": [], \
    "SDmemorySteps": [], \
    }
for dir_name in all_data_directories:
    param = {}
    try:
        raw_params = dir_name.split("_")
        for raw_param in raw_params:
            name, value = raw_param.split("-")
            param[name] = float(value) / 1000.0
    except:
        print("Could not parse '{}'".format(dir_name))
        not_data_directories.append(dir_name)
        continue
    for name in param.keys():
        if param[name] not in all_parameters[name]:
            all_parameters[name].append(param[name])

# Compute the shape for NumPy arrays that will contain all the flight paths.
var_shape = (len(p_goal_range), len(p_social_range), \
    len(p_memory_range), len(sd_goal_range), len(sd_memory_max_range),  \
    len(simulation_types),n_repetitions, 2, n_generations, \
    n_flights_per_generation)
path_shape = (len(p_goal_range), len(p_social_range), \
    len(p_memory_range), len(sd_goal_range), len(sd_memory_max_range), \
    len(simulation_types), n_repetitions, 2, n_generations, \
    n_flights_per_generation, MAX_PATH_LENGTH)

# Create new memory-mapped arrays, and pre-fill them with NaN.
print("Creating files to hold path and efficiency data in...")
x = numpy.memmap(os.path.join(TMP_DIR, "x.dat"), dtype=numpy.float32, \
    mode="w+", shape=path_shape)
x[:] = numpy.NaN
y = numpy.memmap(os.path.join(TMP_DIR, "y.dat"), dtype=numpy.float32, \
    mode="w+", shape=path_shape)
y[:] = numpy.NaN
efficiency = numpy.memmap(os.path.join(TMP_DIR, "efficiency.dat"), \
    dtype=numpy.float64, mode="w+", shape=var_shape)
efficiency[:] = numpy.NaN
print("...New temporary files created!")

# Go through all data directories.
for dir_count, dir_name in enumerate(all_data_directories):
    print("Processing '{}' ({}/{})".format(dir_name, dir_count+1, \
        len(all_data_directories)))
    # Parse the directory name. It will come in this format:
    # Pgoal-10_SDgoal-10_Pcontinuity-10_SDcontinuity-15_Psocial-10_SDsocial- \
    # 40_Pmemory-70_SDmemoryMax-100_SDmemoryMin-50_SDmemorySteps-5
    param = {}
    try:
        raw_params = dir_name.split("_")
        for raw_param in raw_params:
            name, value = raw_param.split("-")
            if name == "SDmemorySteps":
                param[name] = int(value)
            else:
                param[name] = float(value) / 1000.0
    except Exception as e:
        print("Could not parse dir name '{}': {}".format(dir_name, e))
        continue
    # For the precise memory control, 1e-6 is rounded to 0, and will thus have
    # to be corrected after reading.
    if parameter_range == "control_precise-memory":
        param["SDmemoryMax"] = 1e-6
    # Find the index of the parameters in their respective lists.
    pgi = p_goal_range.index(param["Pgoal"])
    psi = p_social_range.index(param["Psocial"])
    pci = p_continuity_range_used.index(param["Pcontinuity"])
    pmi = p_memory_range.index(param["Pmemory"])
    sdgi = sd_goal_range.index(param["SDgoal"])
    sdmmi = sd_memory_max_range.index(param["SDmemoryMax"])
    # Find the sub-directories (these have the actual data in them).
    dir_path = os.path.join(DATADIR, dir_name)
    sub_directory_names = os.listdir(dir_path)
    n_sub_directories = len(sub_directory_names)
    # Loop through all sub-directories.
    for sub_dir_count, sub_dir_name in enumerate(sub_directory_names):
        # Try to parse the name.
        try:
            # Names are in the format "experimental_run-1"
            con, run_nr = sub_dir_name.split("_")
            ci = simulation_types.index(con)
            # Run numbers start at 1, so we subtract 1 to make them compatible
            # with 0-indexing.
            si = run_nr.find("run-") + 4
            run_nr = int(run_nr[si:]) - 1
        except Exception as e:
            print("Could not parse sub-dir name '{}': {}".format( \
                sub_dir_name, e))
            continue
        # Find all files in the sub-directory.
        sub_dir_path = os.path.join(dir_path, sub_dir_name)
        sub_dir_files = os.listdir(sub_dir_path)
        n_sub_dir_files = len(sub_dir_files)
        print("\tProcessing '{}' ({}/{}) with {} files".format(sub_dir_name, \
            sub_dir_count+1, n_sub_directories, n_sub_dir_files))
        # Skip runs that do not have the required number of files.
        if n_sub_dir_files != EXPECTED_N_SUB_DIR_FILES:
            print(("\tExpected {} files, but found {}, so this no further " \
                + "processing of this run will occur").format( \
                EXPECTED_N_SUB_DIR_FILES, n_sub_dir_files))
            with open(LOAD_ERROR_LOG, "a") as f:
                line = [dir_name, sub_dir_name, str(n_sub_dir_files)]
                f.write("\n" + ",".join(line))
            continue
        # Loop through the files.
        for fname in sub_dir_files:
            # Parse the file name.
            try:
                name, ext = os.path.splitext(fname)
                if "efficiency" in name:
                    ftype, gen_nr = name.split("_")
                    flight_nr = "flight-0"
                else:
                    ftype, gen_nr, flight_nr = name.split("_")
            except Exception as e:
                print("\tCould not parse file name '{}': {}".format(fname, e))
                continue
            # Only read files of interest.
            if ftype not in ["xy", "efficiency"]:
                continue
            # Convert string so integers (to be used as indices; all these 
            # start counting at 0).
            si = gen_nr.find("gen-") + 4
            gen_nr = int(gen_nr[si:])
            si = flight_nr.find("flight-") + 7
            flight_nr = int(flight_nr[si:])
            # Construct the path to the file.
            fpath = os.path.join(sub_dir_path, fname)
            # Load the file.
            if ftype == "xy":
                x_, y_ = load_xy_csv(fpath)
                if x_.shape[1] > MAX_PATH_LENGTH:
                    x_ = x_[:,:MAX_PATH_LENGTH]
                    y_ = y_[:,:MAX_PATH_LENGTH]
                ei = x_.shape[1]
                x[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,0,gen_nr,flight_nr,:ei] = \
                    x_[0,:]
                x[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,1,gen_nr,flight_nr,:ei] = \
                    x_[1,:]
                y[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,0,gen_nr,flight_nr,:ei] = \
                    y_[0,:]
                y[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,1,gen_nr,flight_nr,:ei] = \
                    y_[1,:]
            elif ftype == "efficiency":
                efficiency_ = load_efficiency_csv(fpath)
                if efficiency_.shape[1] > n_flights_per_generation:
                    efficiency_ = efficiency_[:,:n_flights_per_generation]
                ei = efficiency_.shape[1]
                efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,0,gen_nr,:] = \
                    efficiency_[0,:ei]
                efficiency[pgi,psi,pmi,sdgi,sdmmi,ci,run_nr,1,gen_nr,:] = \
                    efficiency_[1,:ei]
            else:
                print("\tNo loading implemented for file type '{}'".format( \
                    ftype))
