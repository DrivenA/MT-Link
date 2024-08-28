import torch


def pad_sequences(sequences, max_length, pad_value=0):
    padded_sequences = [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    return padded_sequences, lengths

def create_batch(batch_samples, device):  
    dense_loc = [sample['dense_traj_locID'] for sample in batch_samples]
    dense_tim = [sample['dense_traj_timID'] for sample in batch_samples]
    dense_tims = [sample['dense_traj_timStamp'] for sample in batch_samples]
    sparse_loc = [sample['sparse_traj_locID'] for sample in batch_samples]
    sparse_tim = [sample['sparse_traj_timID'] for sample in batch_samples]
    sparse_tims = [sample['sparse_traj_timStamp'] for sample in batch_samples]
    labels = [sample['IsPositive'] for sample in batch_samples]
    max_length_dense = max(len(seq) for seq in dense_loc)
    max_length_sparse = max(len(seq) for seq in sparse_loc)
    dense_loc_padded, dense_len = pad_sequences(dense_loc, max_length_dense)
    dense_tim_padded, dense_len_tim = pad_sequences(dense_tim, max_length_dense)
    dense_tims_padded, dense_len_tims = pad_sequences(dense_tims, max_length_dense)
    sparse_loc_padded, sparse_len = pad_sequences(sparse_loc, max_length_sparse)
    sparse_tim_padded, sparse_len_tim = pad_sequences(sparse_tim, max_length_sparse)
    sparse_tims_padded, sparse_len_tims = pad_sequences(sparse_tims, max_length_sparse)   
    assert dense_len == dense_len_tim, "Dense lengths mismatch"
    assert sparse_len == sparse_len_tim, "Sparse lengths mismatch"
    assert dense_len == dense_len_tims, "Dense lengths mismatch"
    assert sparse_len == sparse_len_tims, "Sparse lengths mismatch"
    dense_loc_tensor = torch.tensor(dense_loc_padded, dtype=torch.long).to(device)
    dense_tim_tensor = torch.tensor(dense_tim_padded, dtype=torch.long).to(device)
    dense_tims_tensor = torch.tensor(dense_tims_padded, dtype=torch.long).to(device)
    sparse_loc_tensor = torch.tensor(sparse_loc_padded, dtype=torch.long).to(device)
    sparse_tim_tensor = torch.tensor(sparse_tim_padded, dtype=torch.long).to(device)
    sparse_tims_tensor = torch.tensor(sparse_tims_padded, dtype=torch.long).to(device)
    dense_len_tensor = torch.tensor(dense_len, dtype=torch.long).to(device)
    sparse_len_tensor = torch.tensor(sparse_len, dtype=torch.long).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return dense_loc_tensor, dense_tim_tensor, dense_tims_tensor, dense_len_tensor, sparse_loc_tensor, sparse_tim_tensor, sparse_tims_tensor, sparse_len_tensor, labels

    