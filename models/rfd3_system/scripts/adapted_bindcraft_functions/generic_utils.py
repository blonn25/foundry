"""Adapted BindCraft generic utilities for rfd3_system.

Adapted from BindCraft ``functions/generic_utils.py`` at commit
``b971db42ba6e091afab63ccb30ae02215150a990``.

Only utility functions useful outside the full BindCraft design campaign are
active here. BindCraft-specific campaign orchestration functions remain visible
below as commented-out references.
"""

from __future__ import annotations


# Read the pdb file and filter relevant lines
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK'))]

    # Write the cleaned lines back to the original file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)


# calculate averages for statistics
def calculate_averages(statistics, handle_aa=False):
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(1, 6):  # assumes models are numbered 1 through 5
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it separately, do so
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid, initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages


# filter designs based on feature thresholds
def check_filters(mpnn_data, design_labels, filters):
    # check mpnn_data against labels
    mpnn_dict = {label: value for label, value in zip(design_labels, mpnn_data)}

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label == 'Average_InterfaceAAs' or label == '1_InterfaceAAs' or label == '2_InterfaceAAs' or label == '3_InterfaceAAs' or label == '4_InterfaceAAs' or label == '5_InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if mpnn_dict.get(label) is None:
                    continue
                if aa in mpnn_dict[label]:
                    aa_value = mpnn_dict[label][aa]
                    if "min" in aa_conditions and aa_value < aa_conditions["min"]:
                        unmet_conditions.append(f"{label}_{aa}")
                    if "max" in aa_conditions and aa_value > aa_conditions["max"]:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            value = mpnn_dict.get(label)
            if value is None:
                continue
            if "min" in conditions and value < conditions["min"]:
                unmet_conditions.append(label)
            if "max" in conditions and value > conditions["max"]:
                unmet_conditions.append(label)

    if unmet_conditions:
        return False, unmet_conditions

    return True, []


# DROPPED: BindCraft-specific dataframe schema for full campaigns.
# def generate_dataframe_labels():
#     ...

# DROPPED: BindCraft-specific directory layout creation.
# def generate_directories(design_path):
#     ...

# DROPPED: BindCraft-specific failure CSV initialization.
# def generate_filter_pass_csv(failure_csv, filter_json):
#     ...

# DROPPED: BindCraft-specific failure CSV mutation.
# def update_failures(failure_csv, failure_column_or_dict):
#     ...

# DROPPED: BindCraft-specific trajectory stopping rule.
# def check_n_trajectories(design_paths, advanced_settings):
#     ...

# DROPPED: BindCraft-specific accepted-design ranking and copying.
# def check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels):
#     ...

# DROPPED: BindCraft-specific helicity sampling.
# def load_helicity(advanced_settings):
#     ...

# DROPPED: BindCraft-specific JAX GPU check.
# def check_jax_gpu():
#     ...

# DROPPED: BindCraft-specific CLI settings validation.
# def perform_input_check(args):
#     ...

# DROPPED: BindCraft-specific advanced settings defaults.
# def perform_advanced_settings_check(advanced_settings, bindcraft_folder):
#     ...

# DROPPED: BindCraft-specific JSON loading policy.
# def load_json_settings(settings_json, filters_json, advanced_json):
#     ...

# DROPPED: BindCraft-specific AlphaFold2 model selection.
# def load_af2_models(af_multimer_setting):
#     ...

# DROPPED: BindCraft-specific CSV creation.
# def create_dataframe(csv_file, columns):
#     ...

# DROPPED: BindCraft-specific CSV insertion.
# def insert_data(csv_file, data_array):
#     ...

# DROPPED: BindCraft-specific FASTA output into BindCraft design paths.
# def save_fasta(design_name, sequence, design_paths):
#     ...

# DROPPED: BindCraft-specific archive cleanup.
# def zip_and_empty_folder(folder_path, extension):
#     ...
