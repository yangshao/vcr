import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._h5_gen = None

    def __getitem__(self, index):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(index)

    def _get_generator(self):
        with h5py.File( self.h5_path, 'r') as record:
            index = yield
            while True:
                data=record[str(index)]['data'].value
                target=record[str(index)]['target'].value
                index = yield data, target

    def __len__(self):
        with h5py.File(self.h5_path,'r') as record:
            return len(record)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # # --
    # Make data if not there

    if not os.path.exists('test.h5'):
        print('making test.h5')
        f = h5py.File('test.h5')
        for i in range(10000):
            f['%s/data' % i] = np.random.uniform(0, 1, (1024, 1024))
            f['%s/target' % i] = np.random.choice(10)
        f.close()
        print('done')

    # Runs correctly
    # dataloader = torch.utils.data.DataLoader(
    #     H5Dataset('test.h5'),
    #     batch_size=4,
    #     num_workers=0,
    #     shuffle=True
    # )
    #
    # count1=0
    # for i, (data, target) in enumerate(dataloader):
    #     # print(data.shape)
    #     count1+=target
    # print('count1 is equal to: \n{}'.format(count1.sum()))
        # if i > 10:
        #     break

    # Also runs correctly
    dataloader = torch.utils.data.DataLoader(
        H5Dataset('test.h5'),
        batch_size=4,
        num_workers=4,
        shuffle=True
    )

    count2=0
    for i, (data, target) in enumerate(dataloader):
        # print(data.shape)
        # print(target.shape)
        count2+=target
        # if i > 10:
        #     break
    print('count2 is equal to: \n{}'.format(count2.sum()))



