import pandas as pd
import numpy as np

from data_store import LOCAL_DATA_DIR
import ideal_meta

class S2PDataProvider(object):

    def __init__(self, inputs_list, targets_list, timestamps_list, batch_size, input_window_size,
                 output_window_size=1, house_appliances=None, shuffle_order=True, rng=None, inputs_transform=None,
                 targets_transform=None, sample_rate=8, balance_on_off=False, on_power_threshold=5):
        '''
        :param inputs_list: list of aggregate data for each household
        :param targets_list: list of appliance data for each household, each entry with shape (time, appliance)
        :param timestamps_list: list of timestamps for each household
        :param batch_size: Number of data points in each batch
        :param input_window_size: number of time steps taken as input
        :param house_appliances: boolean array specifying appliances in each house (house, appliance)
        :param shuffle_order: Boolean stating whether data will be shuffled
        :param rng: numpy random number generator
        :return:
        '''

        assert len(inputs_list) == len(targets_list) and len(inputs_list) == len(timestamps_list), \
            'inputs, targets and timestamps must have same number of houses'

        self.num_houses = len(inputs_list)
        self.num_appliances = 1 if house_appliances is None else house_appliances.shape[1]
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.timestamps_list = timestamps_list

        self.inputs_transform = inputs_transform
        self.targets_transform = targets_transform

        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.input_offset = input_window_size//2
        self.output_offset = output_window_size//2
        self.house_appliances = house_appliances
        self.shuffle_order = shuffle_order
        self.balance_on_off = balance_on_off
        
        self.on_power_threshold = on_power_threshold

        usable_indices_overlap_list = []
        usable_indices_no_overlap_list = []
        
        for timestamps in timestamps_list:
            usable_indices_array, usable_indices_no_overlap = usable_indices(timestamps, input_window_size, output_window_size)
            usable_indices_overlap_list.append(usable_indices_array)
            usable_indices_no_overlap_list.append(usable_indices_no_overlap)
            
            self.indices = np.hstack(usable_indices_no_overlap_list)
            self.indices_overlap = np.hstack(usable_indices_overlap_list)
            self.indices_houses = np.hstack([idx * np.ones(len(indices), dtype=int) 
                 for idx, indices in enumerate(usable_indices_no_overlap_list)])

        # balance the number of training samples that have appliance on and off 
        # by repeating the ons. We assume there are more offs than ons.
        
        if balance_on_off:
            off_indices_list = []
            on_indices_list = []
            
            off_indices_houses_list = []
            on_indices_houses_list = []
            
            for idx, indices in enumerate(usable_indices_overlap_list):
                
                # check if each input window contains an on
                target_on = (targets_list[idx] >= on_power_threshold).astype(np.int8)
                input_on = np.convolve(target_on, np.ones(input_window_size), mode='same')
                
                on_indices = np.nonzero(input_on)[0]
                on_indices = np.intersect1d(on_indices, indices, assume_unique=True)
                
                off_indices = np.nonzero(np.logical_not(input_on))[0]
                off_indices = np.intersect1d(off_indices, indices, assume_unique=True)
                
                on_indices_list.append(on_indices)
                off_indices_list.append(off_indices)
                
                off_indices_houses_list.append(idx * np.ones(len(off_indices), dtype=int))
                on_indices_houses_list.append(idx * np.ones(len(on_indices), dtype=int))
                  
            self.on_indices = np.hstack(on_indices_list)
            self.on_indices_houses = np.hstack(on_indices_houses_list)
            
            self.off_indices = np.hstack(off_indices_list)
            self.off_indices_houses = np.hstack(off_indices_houses_list)
            
            # repeat on indices to match size of off indices
            reps = self.off_indices.size // self.on_indices.size
            self.on_indices = np.tile(self.on_indices, reps)
            self.on_indices_houses = np.tile(self.on_indices_houses, reps)
            
            self.num_batches = int(np.ceil((self.on_indices.size * 2) // batch_size))
            self.final_batch_size = (self.on_indices.size * 2) % batch_size
        else:
            
            self.num_batches = int(np.ceil(self.indices.size / batch_size))
            self.final_batch_size = self.indices.size % batch_size

        if self.final_batch_size == 0:
            self.final_batch_size = batch_size
            
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.new_epoch()

    def new_epoch(self):
        self.current_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def shuffle(self):

        if self.balance_on_off:
            new_order = self.rng.permutation(self.on_indices.size)
            self.on_indices = self.on_indices[new_order]
            self.on_indices_houses = self.on_indices_houses[new_order]
            
            new_order = self.rng.permutation(self.off_indices.size)
            self.off_indices = self.off_indices[new_order]
            self.off_indices_houses = self.off_indices_houses[new_order]
        else:
            new_order = self.rng.permutation(self.indices.size)
            self.indices = self.indices[new_order]
            self.indices_houses = self.indices_houses[new_order]

    def __iter__(self):
        return self
    
    def next(self):

        if self.current_batch >= self.num_batches:
            self.new_epoch()
            raise StopIteration()
            
        # if we are on last batch, set smaller batch size
        if self.current_batch == (self.num_batches - 1):
            batch_size = self.final_batch_size
        else:
            batch_size = self.batch_size
        
        if self.balance_on_off:
            batch_slice = slice(self.current_batch * (batch_size//2),
                                (self.current_batch + 1) * (batch_size//2))
            
            on_houses_batch = self.on_indices_houses[batch_slice]
        
            inputs_batch_on = np.array(
            [self.inputs_list[on_houses_batch[i]][idx-self.input_offset:idx+self.input_offset+1]
               for i, idx in enumerate(self.on_indices[batch_slice])], dtype=np.float32)
        
            targets_batch_on = np.array([
                self.targets_list[on_houses_batch[i]]
                [idx-self.output_offset:idx+self.output_offset+1]
                 for i, idx in enumerate(self.on_indices[batch_slice])], dtype=np.float32)
            
            off_houses_batch = self.off_indices_houses[batch_slice]
            
            inputs_batch_off = np.array([
                self.inputs_list[off_houses_batch[i]]
                [idx-self.input_offset:idx+self.input_offset+1]
               for i, idx in enumerate(self.off_indices[batch_slice])], dtype=np.float32)
            
            targets_batch_off = np.array([
                self.targets_list[off_houses_batch[i]]
                [idx-self.output_offset:idx+self.output_offset+1]
               for i, idx in enumerate(self.off_indices[batch_slice])], dtype=np.float32)
            
            inputs_batch = np.concatenate([inputs_batch_on, inputs_batch_off])
            targets_batch = np.concatenate([targets_batch_on, targets_batch_off])
        else:
            batch_slice = slice(self.current_batch * batch_size,
                                (self.current_batch + 1) * batch_size)
            
            houses_batch = self.indices_houses[batch_slice]
        
            inputs_batch = np.array(
            [self.inputs_list[houses_batch[i]][idx-self.input_offset:idx+self.input_offset+1]
               for i, idx in enumerate(self.indices[batch_slice])], dtype=np.float32)
        
            targets_batch = np.array(
            [self.targets_list[houses_batch[i]][idx-self.output_offset:idx+self.output_offset+1]
                          for i, idx in enumerate(self.indices[batch_slice])], dtype=np.float32)
            
            
        if self.inputs_transform is not None:
            inputs_batch = self.inputs_transform(inputs_batch)

        if self.targets_transform is not None:
            targets_batch = self.targets_transform(targets_batch)
        
        self.current_batch += 1
        
        if self.house_appliances is None:
            return inputs_batch, targets_batch
        else:
            appliances_batch = self.house_appliances[houses_batch]
            return [inputs_batch, appliances_batch], targets_batch,
            # shapes (batch_size, input_window),(batch_size, output_window, num_appliances),(batch_size, num_appliances)

    def generator(self):
        while True:
            for batch in self:
                yield batch
    
    def __next__(self):
        return self.next()
    
    
class S2SSyntheticDataProvider:
    
    def __init__(self, inputs, targets, targets_mask, appliance, distractor_appliances,
                 batch_size, shuffle_order=True, scale_activations=False):
        self.aggregate_scaling_factor = 0.93
        self.synthetic_fraction = 0.5
        self.scaling_range = 0.1
        
        self.batch_size = batch_size
        self.appliance = appliance
        self.distractor_appliances = distractor_appliances
        self.shuffle_order = shuffle_order
        self.scale_activations = scale_activations
        self.inputs = inputs
        self.targets = targets
        self.targets_mask = targets_mask
        
        self.all_appliances = [appliance] + distractor_appliances
        
        appliance_stats = pd.read_csv('../ideal/appliance_stats.csv')
        appliance_stats = appliance_stats[appliance_stats.appliancetype==appliance]
        self.mean_on_power = float(appliance_stats.mean_on_power)
        
        self.appliance_activations = [
            np.load(LOCAL_DATA_DIR + '/nilm/s2s_{0}_train_activations.npy'.format(appliance))
                    for appliance in self.all_appliances]
        
        self.appliance_activation_lengths = [
          np.load(LOCAL_DATA_DIR + '/nilm/s2s_{0}_train_activation_length.npy'.format(appliance))
                    for appliance in self.all_appliances]
        
        # preprocess data
        self.inputs = np.nan_to_num(self.inputs)
        self.targets = np.nan_to_num(self.targets)
        np.putmask(self.targets, self.targets_mask, 0)
        self.targets_mask = ~self.targets_mask  # masked targets are now zero
        
        self.distractor_activations = self.appliance_activations[1:]
        self.distractor_activation_lengths = self.appliance_activation_lengths[1:]
        
        self.appliance_activations = self.appliance_activations[0]
        self.appliance_activation_lengths = self.appliance_activation_lengths[0]
        
        self.input_window_size = self.inputs.shape[1]
        self.output_window_size = self.targets.shape[1]
        self.offset = (self.input_window_size - self.output_window_size) // 2
        
        # calculate number of batches
        self.real_batch_size = int(batch_size * (1 - self.synthetic_fraction))
        self.num_batches = int(np.ceil(self.inputs.shape[0] / self.real_batch_size))
        self.final_real_batch_size = self.inputs.shape[0] % self.real_batch_size
        if self.final_real_batch_size == 0:
            self.final_real_batch_size = self.real_batch_size
            
        self.indices = np.arange(self.inputs.shape[0])
        
        self.new_epoch()
            
    def new_epoch(self):
        self.current_batch = 0
        if self.shuffle_order:
            self.shuffle()
            
    def shuffle(self):
        self.indices = np.random.permutation(self.indices.size)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.current_batch >= self.num_batches:
            self.new_epoch()
            raise StopIteration()
        
        
                # if we are on last batch, set smaller batch size
        if self.current_batch == (self.current_batch - 1):
            real_batch_size = self.final_real_batch_size
        else:
            real_batch_size = self.real_batch_size
        
        synthetic_batch_size = self.batch_size - real_batch_size
        
        real_batch_indices = self.indices[self.current_batch * self.real_batch_size:
                          self.current_batch * self.real_batch_size + real_batch_size]
        
        real_batch_inputs = self.inputs[real_batch_indices]
        real_batch_targets = self.targets[real_batch_indices]
        real_batch_targets_mask = self.targets_mask[real_batch_indices]
        
        # generate sythetic data
        synthetic_inputs, synthetic_targets, synthetic_targets_mask \
                    = self.generate_synthetic_batch(synthetic_batch_size)
            
        synthetic_inputs *= self.aggregate_scaling_factor
        
        # merge
        batch_inputs = np.concatenate([real_batch_inputs, synthetic_inputs])
        batch_targets = np.concatenate([real_batch_targets, synthetic_targets])
        batch_targets_mask = np.concatenate([real_batch_targets_mask, synthetic_targets_mask])
        
        # normalise
        batch_inputs = ((batch_inputs - batch_inputs.mean(axis=1).reshape((-1,1))) 
                        / ideal_meta.mains_std)
        batch_targets = batch_targets / self.mean_on_power
        
        self.current_batch += 1
        
        return [batch_inputs, batch_targets_mask], batch_targets
        
        
    def generate_synthetic_batch(self, sythetic_batch_size):
        synthetic_inputs = np.zeros((sythetic_batch_size, self.input_window_size))
        appliance_present = np.nonzero(np.random.random(sythetic_batch_size) < 0.5)[0]
        num_appliance_present = appliance_present.size
        activation_indices = np.random.randint(self.appliance_activations.shape[0],
                                                         size=(num_appliance_present,))
        activation_lengths = self.appliance_activation_lengths[activation_indices]
        # offset between start of input window and start of appliance activation
        activation_offsets = np.floor(np.random.random(num_appliance_present) * (
            self.input_window_size + activation_lengths) - activation_lengths).astype(np.int64)
        
        inputs_activation_start = np.maximum(0, activation_offsets)
        inputs_activation_end = np.minimum(self.input_window_size,
                                           activation_offsets + activation_lengths)
        
        activation_start = np.maximum(0, -activation_offsets)
        activation_end = np.minimum(activation_lengths,
                                    self.input_window_size - activation_offsets)
        
        for i in range(num_appliance_present):
            synthetic_inputs[appliance_present[i],
                             inputs_activation_start[i]:inputs_activation_end[i]] \
                = self.appliance_activations[activation_indices[i],
                                             activation_start[i]:activation_end[i]]
                
        if self.scale_activations:
            scaling_factors = np.random.random((sythetic_batch_size, 1)) * self.scaling_range*2 + 1
            synthetic_inputs *= scaling_factors
        
        # create targets
        synthetic_targets = synthetic_inputs[:, self.offset:-self.offset].copy()
        
        # add distractor appliances
        for appliance_i, activations in enumerate(self.distractor_activations):
            appliance_present = np.nonzero(np.random.random(sythetic_batch_size) < 0.25)[0]
            num_appliance_present = appliance_present.size
            activation_indices = np.random.randint(activations.shape[0],
                                                             size=(num_appliance_present,))
            activation_lengths \
                = self.distractor_activation_lengths[appliance_i][activation_indices]
                
            # offset between start of input window and start of appliance activation
            activation_offsets = np.floor(np.random.random(num_appliance_present) 
                    * (self.input_window_size + activation_lengths)
                    - activation_lengths).astype(np.int64)

            inputs_activation_start = np.maximum(0, activation_offsets)
            inputs_activation_end = np.minimum(self.input_window_size,
                                               activation_offsets + activation_lengths)

            activation_start = np.maximum(0, -activation_offsets)
            activation_end = np.minimum(activation_lengths,
                                        self.input_window_size - activation_offsets)
            
            scaling_factors = np.random.random(num_appliance_present) * self.scaling_range*2 + 1
            
            for i in range(num_appliance_present):
                synthetic_inputs[appliance_present[i],
                                 inputs_activation_start[i]:inputs_activation_end[i]] \
                    += activations[activation_indices[i],
                                   activation_start[i]:activation_end[i]] * scaling_factors[i]
        # mask should be all zeros
        synthetic_targets_mask = np.ones_like(synthetic_targets)
#         if  np.isnan(synthetic_inputs).any():
#             import pdb;pdb.set_trace()
        return synthetic_inputs, synthetic_targets, synthetic_targets_mask
    
    def generator(self):
        while True:
            for batch in self:
                yield batch


def usable_indices(timestamps, window_length, output_window_length=1, threshold=8):
    if len(timestamps) == 0:
        indices = np.array([], dtype=(np.int32))
    else:
        offset = int(window_length//2 + 1)
        deltas = timestamps[1:] - timestamps[:-1]
        gaps = (deltas > threshold) | (deltas < 0)
        indices = np.logical_not(np.convolve(np.ones(offset), gaps, mode='same'))
        indices = np.nonzero(indices)[0]
        indices = indices[np.nonzero((indices >= (offset-1)) 
                                     & (indices <= (len(timestamps) - offset)))]
        #if output_window_length > 1

    return indices, indices

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

