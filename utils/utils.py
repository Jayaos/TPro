import os
import pickle


def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def safe_save_pickle(filename, obj):
    temp_filename = filename + ".tmp"

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(temp_filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())  # ensure all data is written to disk
        os.replace(temp_filename, filename)  # atomic replace
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)