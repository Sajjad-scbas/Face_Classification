import numpy as np
import pickle 
import torch

from tqdm import tqdm

from model.model_V2 import SwinTransformer
from torch.utils.data import DataLoader


def inference(model_name, file_path):
    # Prepare the test data
    nb_images = 10130
    height = 56
    width = 56 
    channels = 3
    with open(f"{file_path}/db_test.raw", 'rb') as file:
        test_data = file.read()
    test_data = np.frombuffer(test_data, dtype=np.uint8).reshape(nb_images, height, width, channels).astype(np.float32)

    # Prepare the model 
    with open(f'{model_name}_params.pkl', 'rb') as file :
        params = pickle.load(file)
    device = params['device']
    dtype = torch.float32

    model = SwinTransformer(params).to(dtype = dtype, device= device)
    model.load_state_dict(torch.load(f"{model_name}.pt"))

    test_data = torch.from_numpy(test_data).to(device=device, dtype=dtype)
    #to overcome the GPU memory problem
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    predictions = torch.empty(nb_images, dtype=torch.uint8)
    for idx, data in enumerate(tqdm(test_dataloader)):
        _, predicted = torch.max(model(data), 1)
        predictions[idx*batch_size:idx*batch_size+len(data)] = predicted

    np.savetxt("predictions.txt", predictions.detach().cpu().numpy(), fmt='%d')

    print("Prediction Saved!")


if __name__=="__main__":
    model_name = "model_N5"
    file_path = '/home/sajjad/Desktop/Face_Classification/data'
    inference(model_name, file_path)
    print("Done")
