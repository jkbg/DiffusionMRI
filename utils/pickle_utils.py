import os
import pickle


def load_results(file_path):
    results = []
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as input:
            while True:
                try:
                    results.append(pickle.load(input))
                except EOFError:
                    break
    return results


def save_gridsearch_result(result, file_path):
    results = load_results(file_path)
    results.append(result)
    with open(file_path, 'wb') as output:
        for result in results:
            pickle.dump(result, output, pickle.HIGHEST_PROTOCOL)