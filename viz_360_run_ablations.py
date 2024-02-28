import os, sys, time, json, tyro
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy import arctan2

@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    results_dir: str='/scratch/vineeth.bhat/vin-experiments/360_runs/'
    device: str='cuda'
    plot_save_dir: str='/scratch/vineeth.bhat/vin-experiments/360_runs/'
    parameter: str='peak_gpu_usage'
    num_min_marked: int = 5

def calculate_rmse(values_list):
    np_values_list = np.array(values_list)
    rmse_array = np.sqrt(np.mean(np_values_list**2, axis=1))
    return rmse_array

if __name__=="__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    if largs.parameter in ["peak_gpu_usage", "total_time", "memory_usage"]: 
        file_names = []
        list_of_numbers_in_file_names = []
        values = []

        for file_name in os.listdir(largs.results_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(largs.results_dir, file_name)
                numbers_in_name = os.path.splitext(file_name)[0].split("results_")[1].split("_")
                list_of_numbers_in_file_names.append(numbers_in_name)

                formatted_numbers = ["{:0.3f}".format(float(num)) if num.replace(".", "").isdigit() 
                                    else num for num in numbers_in_name]

                file_names.append("  ".join(formatted_numbers))

                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    peak_gpu_usage = data.get(largs.parameter, None)
                    if peak_gpu_usage is not None:
                        values.append(peak_gpu_usage)
                    else:
                        raise

        data_tuples = sorted(zip(list_of_numbers_in_file_names, file_names, values), key=lambda x: x[0])
        list_of_numbers_in_file_names, file_names, values = zip(*data_tuples)

        plt.figure(figsize=(16, 7))
        plt.plot(file_names, values, linestyle='-', color='r')
        plt.bar(file_names, values)

        # Identify indices of minimum values
        min_indices = np.argsort(values)[:largs.num_min_marked]

        # Mark labels corresponding to minimum values in blue
        for idx in min_indices:
            plt.text(idx, values[idx], "  ||==||", color='blue', rotation=90, ha='center', va='bottom')

        plt.xlabel('File Name')
        plt.ylabel(f'Parameter {largs.parameter}')
        plt.title(f'Ablations over {largs.parameter}')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        # Save the plot without displaying it
        plt.savefig(os.path.join(largs.plot_save_dir, f"{largs.parameter}.png"))

    elif largs.parameter in ["translation_rmses", "rotation_rmses"]:
        file_names = []
        list_of_numbers_in_file_names = []
        list_of_values = []

        for file_name in os.listdir(largs.results_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(largs.results_dir, file_name)
                numbers_in_name = os.path.splitext(file_name)[0].split("results_")[1].split("_")
                list_of_numbers_in_file_names.append(numbers_in_name)

                formatted_numbers = ["{:0.3f}".format(float(num)) if num.replace(".", "").isdigit() 
                                    else num for num in numbers_in_name]

                file_names.append("  ".join(formatted_numbers))

                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                    values = data.get(largs.parameter, None)
                    if values is not None:
                        list_of_values.append(values)
                    else:
                        raise

        num_indices = len(list_of_values[0])  # Assuming all lists have the same length

        # Create subplots
        fig, axes = plt.subplots(num_indices + 1, 1, figsize=(16, 7*(num_indices + 1)), sharex=True)

        data_tuples = sorted(zip(list_of_numbers_in_file_names, file_names, list_of_values), key=lambda x: x[0])
        list_of_numbers_in_file_names, file_names, list_of_values = zip(*data_tuples)

        for idx in range(num_indices):
            subplot_values = [values[idx] for values in list_of_values]

            # Plot subplot
            axes[idx].plot(file_names, subplot_values, linestyle='-', color='r')
            axes[idx].bar(file_names, subplot_values)
            axes[idx].set_ylabel(f'{largs.parameter} - Pose {idx + 1}')

        # Calculate and plot RMSE subplot
        rmse = calculate_rmse(list_of_values)
        axes[num_indices].bar(file_names, rmse)
        axes[num_indices].plot(file_names, rmse, linestyle='-', color='r')
        axes[num_indices].set_ylabel('RMSE of all Poses')

        # Identify indices of minimum RMSE values
        min_rmse_indices = np.argsort(rmse)[:largs.num_min_marked]

        # Mark labels corresponding to minimum RMSE values in blue
        for idx in min_rmse_indices:
            axes[num_indices].text(idx, rmse[idx], "  ||==||", color='blue', rotation=90, ha='center', va='bottom')

        plt.xlabel('File Name')
        plt.suptitle(f'Ablations over {largs.parameter}')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(largs.plot_save_dir, f"{largs.parameter}.png"))
