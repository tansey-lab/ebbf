import numpy as np
import pyarrow.dataset as ds


def batch_generator(dataset: ds.Dataset, cols: list[str], batch_size: int):
    n_cols = len(cols)
    batches = dataset.to_batches(batch_size=batch_size, columns=cols)

    curr_datas = []
    curr_size = 0

    for batch in batches:
        data = [np.array(batch[col]) for col in cols]
        curr_datas.append(data)
        curr_size += data[0].shape[0]

        if curr_size >= batch_size:
            data = tuple(np.concatenate([x[i] for x in curr_datas]) for i in range(n_cols))
            curr_datas = []
            curr_size = 0
            yield data

    if curr_size > 0:
        data = tuple(np.concatenate([x[i] for x in curr_datas]) for i in range(n_cols))
        yield data
