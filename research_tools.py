import os
import numpy as np

def cached_action(action, cache_location, save_names=[]):
    should_act = True
    if len(save_names) != 0:
        i = 0
        while should_act and i < len(save_names):
            should_act = not os.path.isfile(cache_location + save_names[i] +
                    '.npy')
            i += 1
    else:
        should_act = not os.path.isfile(cache_location + '.npy')

    if should_act:
        result = action()
        if len(save_names) > 0:
            for i, save_name in enumerate(save_names):
                np.save(cache_location + save_name + '.npy', result[i])
        else:
            np.save(cache_location + '.npy', result)

        return result
    else:
        if len(save_names) > 0:
            loaded_result = []
            for i, save_name in enumerate(save_names):
                loaded_result.append(np.load(cache_location + save_name +
                    '.npy'))

            return loaded_result
        else:
            return np.load(cache_location + '.npy')
