import numpy as np
from tools import *
from tqdm import tqdm
import sys
sys.path.insert(1, '../DeepSDF2')
import deep_sdf

# load our autodecoder shapecodes
# model_datetime = '11012022_133638'  # chairs
# model_datetime = '11082022_200446' # planes

# shapecode, shapecode_epoch, shape_indices = get_shapecode(model_datetime, n_shapes=(4045-500))  #6778 or 4045
# decoder, _ = get_decoder(model_datetime, decoder_epoch=shapecode_epoch, parallel=True) 

# print(model_datetime, shapecode_epoch)
# print(shapecode.shape)

# std = np.std(np.asarray(shapecode))
# mean = np.mean(np.asarray(shapecode))

# print(std, mean)

# if not os.path.isdir('./generated_gaussian_our_autodecoder(planes)/'):
#     os.mkdir('./generated_gaussian_our_autodecoder(planes)/')

# for i in tqdm(range(500)):
#     if os.path.isfile(f'./generated_gaussian_our_autodecoder(planes)/{i}.npz'):
#         continue
#     code_t = np.random.normal(mean, std, 256)
#     verts, faces = reconstruct_normalized_shape(decoder, torch.from_numpy(code_t).float())
#     np.savez(f'./generated_gaussian_our_autodecoder(planes)/{i}', denoised_code=code_t, verts=verts, faces=faces)
    

    
    
# load deepsdf shapecodes
model_datetime = 'planes'
shapecode, shapecode_epoch = get_deep_sdf_shapecodes(experiment_directory=f"../DeepSDF2/examples/{model_datetime}", checkpoint="latest")
decoder, _ = get_deep_sdf_decoder(experiment_directory=f"../DeepSDF2/examples/{model_datetime}", checkpoint="latest", parallel=True)

print(model_datetime, shapecode_epoch)
print(shapecode.shape)

std = np.std(np.asarray(shapecode))
mean = np.mean(np.asarray(shapecode))

print(std, mean)

if not os.path.isdir('./generated_gaussian_deepsdf/'):
    os.mkdir('./generated_gaussian_deepsdf/')

for i in tqdm(range(500)):
    if os.path.isfile(f'./generated_gaussian_deepsdf/{i}.npz'):
        continue
    code_t = np.random.normal(mean, std, 256)
    with torch.no_grad():
        verts, faces = deep_sdf.mesh.create_mesh(
            decoder,
            torch.from_numpy(code_t).float(),
            "",
            N=256,
            max_batch=int(2 ** 18),
            offset=None,
            scale=None,
        )
    np.savez(f'./generated_gaussian_deepsdf/{i}', denoised_code=code_t, verts=verts, faces=faces)

     

