"""
Data reading utilities for the ionerdss package.
This module centralizes all data file reading operations to avoid redundancy.
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any


def read_copy_numbers(sim_dir: str) -> Optional[pd.DataFrame]:
    """
    Read species copy numbers vs time data from a simulation directory.
    
    Parameters:
        sim_dir (str): Path to the simulation directory
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing the data, or None if file not found
    """
    data_file = os.path.join(sim_dir, "DATA", "copy_numbers_time.dat")
    
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found, skipping simulation.")
        return None
    
    df = pd.read_csv(data_file)
    df.rename(columns=lambda x: x.strip(), inplace=True)  # Strip spaces from column names
    
    return df


def parse_complex_line(line: str) -> Tuple[Optional[int], Optional[Dict[str, float]]]:
    """
    Parse a single complex line and return a dictionary with species and counts.
    
    Parameters:
        line (str): A line of text containing complex data
        
    Returns:
        Tuple[Optional[int], Optional[Dict[str, float]]]: 
            A tuple containing the count of complexes and a dictionary mapping species to counts
    """
    match = re.match(r"(\d+)\s+([\w\.\s:]+)", line)
    if not match:
        return None, None

    count = int(match.group(1))  # Number of such complexes
    species_data = match.group(2).split()  # Split species and counts
    species_dict = {}

    for i in range(0, len(species_data), 2):
        species_name = species_data[i].strip(":")
        species_count = int(species_data[i + 1].strip("."))
        species_dict[species_name] = species_dict.get(species_name, 0) + species_count

    return count, species_dict


def read_histogram_complexes(sim_dir: str) -> Dict[str, Any]:
    """
    Read histogram complex data from a simulation directory.
    
    Parameters:
        sim_dir (str): Path to the simulation directory
        
    Returns:
        Dict[str, Any]: Dictionary containing time series and complex data
    """
    data_file = os.path.join(sim_dir, "DATA", "histogram_complexes_time.dat")
    
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found, skipping simulation.")
        return {"time_series": [], "complexes": []}
    
    time_series = []
    all_complexes = []
    
    with open(data_file, "r") as f:
        lines = f.readlines()
    
    current_time = None
    current_complexes = []
    
    for line in lines:
        time_match = re.match(r"Time \(s\): (\d*\.?\d+)", line)
        if time_match:
            if current_time is not None:
                time_series.append(current_time)
                all_complexes.append(current_complexes)
                current_complexes = []
            
            current_time = float(time_match.group(1))
        else:
            count, species_dict = parse_complex_line(line)
            if species_dict:
                current_complexes.append((count, species_dict))
    
    # Add the last time point
    if current_time is not None and current_complexes:
        time_series.append(current_time)
        all_complexes.append(current_complexes)
    
    return {
        "time_series": time_series,
        "complexes": all_complexes
    }


def filter_by_time_frame(data: Dict[str, Any], time_frame: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Filter histogram complex data by time frame.
    
    Parameters:
        data (Dict[str, Any]): Dictionary containing time series and complex data
        time_frame (Optional[Tuple[float, float]]): Time range (start, end) to filter by
        
    Returns:
        Dict[str, Any]: Filtered data dictionary
    """
    if time_frame is None:
        return data
    
    start, end = time_frame
    filtered_indices = [
        i for i, t in enumerate(data["time_series"]) 
        if start <= t <= end
    ]
    
    return {
        "time_series": [data["time_series"][i] for i in filtered_indices],
        "complexes": [data["complexes"][i] for i in filtered_indices]
    }


def parse_transition_lifetime_data(file_path: str, time_frame: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict[int, List[float]]]:
    """
    Parse transition matrix and lifetime data from a file.
    
    Parameters:
        file_path (str): Path to the transition matrix file
        time_frame (Optional[Tuple[float, float]]): Time range (start, end) to consider
        
    Returns:
        Tuple[np.ndarray, Dict[int, List[float]]]: 
            A tuple containing the transition matrix and a dictionary of lifetimes per cluster size
    """
    with open(file_path, "r") as f:
        content = f.read()

    time_blocks = re.split(r"time:\s*", content)[1:]  # Skip the first split which is before the first time entry

    time_data = []

    for block in time_blocks:
        lines = block.strip().splitlines()
        time_val = float(lines[0])
        
        # Parse transition matrix
        tm_start = lines.index("transion matrix for each mol type: ") + 2  # skip the "A" line
        tm_lines = []
        for i in range(tm_start, len(lines)):
            if lines[i].startswith("lifetime for each mol type: "):
                break
            if lines[i].strip():
                tm_lines.append([int(x) for x in lines[i].split()])
        transition_matrix = np.array(tm_lines)

        # Parse lifetimes
        lt_block = lines[i+2:]  # start after "A"
        lifetime = defaultdict(list)
        cluster_size = None
        for line in lt_block:
            if line.startswith("size of the cluster:"):
                cluster_size = int(line.split(":")[1])
            elif cluster_size is not None and line.strip():
                lifetime[cluster_size].extend([float(x) for x in line.strip().split()])

        time_data.append((time_val, transition_matrix, lifetime))

    # Sort by time just in case
    time_data.sort(key=lambda x: x[0])

    if time_frame:
        start, end = time_frame
        # Find the nearest time to 'start' and 'end'
        nearest_start = min(time_data, key=lambda x: abs(x[0] - start))
        nearest_end = min(time_data, key=lambda x: abs(x[0] - end))

        t_start, tm_start, lt_start = nearest_start
        t_end, tm_end, lt_end = nearest_end

        if tm_start is None or tm_end is None:
            raise ValueError("Specified time range not found in data.")

        matrix_delta = tm_end - tm_start

        # Lifetime delta: subtract based on how many entries were already present
        lifetime_delta = defaultdict(list)
        for k in lt_end:
            lt1 = lt_start.get(k, [])
            lt2 = lt_end.get(k, [])
            lifetime_delta[k] = lt2[len(lt1):]  # get only new entries

    else:
        matrix_delta = time_data[-1][1]
        lifetime_delta = time_data[-1][2]

    return matrix_delta, lifetime_delta


def read_transition_matrix(sim_dir: str, time_frame: Optional[Tuple[float, float]] = None) -> Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]:
    """
    Read transition matrix and lifetime data from a simulation directory.
    
    Parameters:
        sim_dir (str): Path to the simulation directory
        time_frame (Optional[Tuple[float, float]]): Time range (start, end) to consider
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]: 
            A tuple containing the transition matrix and lifetime data, or (None, None) if file not found
    """
    file_path = os.path.join(sim_dir, "DATA", "transition_matrix_time.dat")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping simulation.")
        return None, None
    
    return parse_transition_lifetime_data(file_path, time_frame)


def compute_average_assembly_size(complexes: List[Tuple[int, Dict[str, float]]], conditions: List[str]) -> Dict[str, float]:
    """
    Compute the average assembly size for given conditions.

    Parameters:
        complexes (list): List of tuples (count, species_dict) representing each complex.
        conditions (list): List of conditions, e.g., ["A>=2", "A+B>=4"].

    Returns:
        dict: Condition -> average assembly size mapping.
    """
    results = {}

    for condition in conditions:
        species_conditions = condition.split(", ")  # Handle multiple species constraints
        numerator, denominator = 0, 0

        for count, species_dict in complexes:
            valid = True
            total_size = 0

            for cond in species_conditions:
                species_match = re.match(r"(\w+)([>=<]=?|==)(\d+)", cond)
                if not species_match:
                    continue  # Skip invalid conditions

                species, operator, threshold = species_match.groups()
                threshold = int(threshold)
                species_count = species_dict.get(species, 0)

                if operator == ">=" and species_count < threshold:
                    valid = False
                elif operator == ">" and species_count <= threshold:
                    valid = False
                elif operator == "<=" and species_count > threshold:
                    valid = False
                elif operator == "<" and species_count >= threshold:
                    valid = False
                elif operator == "==" and species_count != threshold:
                    valid = False

                total_size += species_count  # Sum the species count

            if valid:
                numerator += count * total_size
                denominator += count

        results[condition] = numerator / denominator if denominator > 0 else 0

    return results


def eval_condition(species_dict: Dict[str, float], condition: str) -> Tuple[bool, str]:
    """
    Evaluates whether a complex meets a condition based on species count.
    
    Parameters:
        species_dict (dict): Dictionary containing species counts in one complex.
        condition (str): A condition string like "B>=3".
    
    Returns:
        Tuple[bool, str]: (True if the complex satisfies the condition, species name)
    """
    species_match = re.match(r"(\w+)([>=<]=?|==)(\d+)", condition)
    if not species_match:
        return False, ""

    species, operator, threshold = species_match.groups()
    threshold = int(threshold)
    
    species_count = species_dict.get(species, 0)  # Get count for the species
    return eval(f"{species_count} {operator} {threshold}"), species


def read_multiple_simulations(sim_dirs: List[str], reader_func: callable, *args, **kwargs) -> List[Any]:
    """
    Read data from multiple simulations using the provided reader function.
    
    Parameters:
        sim_dirs (List[str]): List of simulation directories
        reader_func (callable): Function to read data from a single simulation
        *args, **kwargs: Additional arguments to pass to the reader function
        
    Returns:
        List[Any]: List of data from each simulation, with None for simulations that failed
    """
    results = []
    
    for idx, sim_dir in enumerate(sim_dirs):
        try:
            result = reader_func(sim_dir, *args, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Error reading data from simulation {idx} at {sim_dir}: {e}")
            results.append(None)
    
    return results


class DataIO:
    """
    A class to handle all data input and output for the ionerdss package.
    This centralizes file access and implements caching to avoid repeated file reads.
    """
    
    def __init__(self):
        """
        Initialize a cache for storing data to avoid repeated file reads.
        Also initialize last-modified time storage for cache invalidation.
        """
        self._cache = {}
        self._modified = {}
        
    def clear_cache(self):
        """
        Clear the cached data.
        """
        self._cache = {}
        self._modified = {}
        
    def get_copy_numbers(self, sim_dir: str) -> Optional[pd.DataFrame]:
        """
        Get species copy numbers data from a simulation directory, using cache if available.
        
        Parameters:
            sim_dir (str): Path to the simulation directory
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing the data, or None if file not found
        """
        cache_key = (sim_dir, "copy_numbers")
        data_file = os.path.join(sim_dir, "DATA", "copy_numbers_time.dat")
        mod_time = os.path.getmtime(data_file) # get last modified timestamp
        
        if cache_key in self._cache:
            if self._modified[cache_key] == mod_time:
                return self._cache[cache_key]
            else:
                print("File modified, reloading from disk.")
        
        result = read_copy_numbers(sim_dir)
        if result is not None:
            self._cache[cache_key] = result
            self._modified[cache_key] = mod_time
        return result
    
    def get_histogram_complexes(self, sim_dir: str) -> Dict[str, Any]:
        """
        Get histogram complex data from a simulation directory, using cache if available.
        
        Parameters:
            sim_dir (str): Path to the simulation directory
            
        Returns:
            Dict[str, Any]: Dictionary containing time series and complex data
        """
        cache_key = (sim_dir, "histogram_complexes")
        data_file = os.path.join(sim_dir, "DATA", "histogram_complexes_time.dat")
        mod_time = os.path.getmtime(data_file) # get last modified timestamp
        
        if cache_key in self._cache:
            if self._modified[cache_key] == mod_time:
                return self._cache[cache_key]
            else:
                print("File modified, reloading from disk.")
        
        result = read_histogram_complexes(sim_dir)
        self._cache[cache_key] = result
        self._modified[cache_key] = mod_time
        return result
    
    def get_transition_matrix(self, sim_dir: str, time_frame: Optional[Tuple[float, float]] = None) -> Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]:
        """
        Get transition matrix and lifetime data from a simulation directory, using cache if available.
        
        Parameters:
            sim_dir (str): Path to the simulation directory
            time_frame (Optional[Tuple[float, float]]): Time range (start, end) to consider
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]: 
                A tuple containing the transition matrix and lifetime data, or (None, None) if file not found
        """
        # When time_frame is provided, we need to include it in the cache key
        # but tuples with floats may not hash well, so convert to strings
        time_str = None
        if time_frame is not None:
            time_str = f"{time_frame[0]}-{time_frame[1]}"
        cache_key = (sim_dir, "transition_matrix", time_str)
        data_file = os.path.join(sim_dir, "DATA", "transition_matrix_time.dat")
        mod_time = os.path.getmtime(data_file) # get last modified timestamp
        
        if cache_key in self._cache:
            if self._modified[cache_key] == mod_time:
                return self._cache[cache_key]
            else:
                print("File modified, reloading from disk.")
        
        result = read_transition_matrix(sim_dir, time_frame)
        if result[0] is not None:
            self._cache[cache_key] = result
            self._modified[cache_key] = mod_time
        return result
    
    def get_multiple_copy_numbers(self, sim_dirs: List[str]) -> List[Optional[pd.DataFrame]]:
        """
        Get copy numbers data from multiple simulation directories.
        
        Parameters:
            sim_dirs (List[str]): List of simulation directories
            
        Returns:
            List[Optional[pd.DataFrame]]: List of DataFrames, with None for simulations that failed
        """
        return [self.get_copy_numbers(sim_dir) for sim_dir in sim_dirs]
    
    def get_multiple_histogram_complexes(self, sim_dirs: List[str]) -> List[Dict[str, Any]]:
        """
        Get histogram complex data from multiple simulation directories.
        
        Parameters:
            sim_dirs (List[str]): List of simulation directories
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing time series and complex data
        """
        return [self.get_histogram_complexes(sim_dir) for sim_dir in sim_dirs]
    
    def get_multiple_transition_matrices(self, sim_dirs: List[str], time_frame: Optional[Tuple[float, float]] = None) -> List[Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]]:
        """
        Get transition matrix and lifetime data from multiple simulation directories.
        
        Parameters:
            sim_dirs (List[str]): List of simulation directories
            time_frame (Optional[Tuple[float, float]]): Time range (start, end) to consider
            
        Returns:
            List[Tuple[Optional[np.ndarray], Optional[Dict[int, List[float]]]]]: 
                List of tuples containing transition matrix and lifetime data
        """
        return [self.get_transition_matrix(sim_dir, time_frame) for sim_dir in sim_dirs]
