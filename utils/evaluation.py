import os
import pandas as pd
from tqdm import tqdm
from utils.compute_fid import compute_fid


# move file to different folder based on category
save_dir = 'temp_test'
base_dir = '/home/donaldtrump/haolan/pytorch-CycleGAN-and-pix2pix/pix2pix_results'
# base_dir = 'fid_images/DiT-XL-4-all_cond-512_combined'

# seed = 10
seed = None
sample_dir = base_dir + f'/samples_{seed}' if seed else base_dir + '/samples'
gt_dir = base_dir + '/gt'

print(f'Base directory: {base_dir}')
print("moving files to different folders based on category")
# delete the save_dir if it exists
if os.path.exists(save_dir):
    os.system(f'rm -r {save_dir}')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)

# make directories based on category
os.makedirs(os.path.join(save_dir, 'samples', '0'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'samples', '1'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'samples', '2'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'samples', '3'), exist_ok=True)

os.makedirs(os.path.join(save_dir, 'gt', '0'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'gt', '1'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'gt', '2'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'gt', '3'), exist_ok=True)


df = pd.read_csv('/home/donaldtrump/haolan/msd_dataset/test_file_room_num_pair.csv')


# move files to different folders based on category, each category only receives 100 samples
category_count = [0, 0, 0, 0]
for i in tqdm(os.listdir(sample_dir)):
    if '_s' in i:
        category = df[df['source'] == i.replace('_s', '')]['category'].values[0]
    else:
        category = df[df['source'] == i]['category'].values[0]
    # copy sample file to corresponding category folder
    os.system(f'cp {os.path.join(sample_dir, i)} {os.path.join(save_dir, "samples", str(category))}')
    # category_count[category] += 1
    # if category_count[category] == 100:
    #     continue
category_count = [0, 0, 0, 0]
for i in tqdm(os.listdir(gt_dir)):
    if '_s' in i:
        category = df[df['source'] == i.replace('_s', '')]['category'].values[0]
    else:
        category = df[df['source'] == i]['category'].values[0]
    # copy sample file to corresponding category folder
    os.system(f'cp {os.path.join(gt_dir, i)} {os.path.join(save_dir, "gt", str(category))}')
    # category_count[category] += 1
    # if category_count[category] == 100:
    #     continue



print("Computing FID scores")
fid_scores = []
# compute fid on all images
gt_dir = os.path.join(base_dir, 'gt')
pred_dir = os.path.join(base_dir, f'samples_{seed}' if seed else 'samples')
fid_score = compute_fid(gt_dir, pred_dir)
fid_scores.append(fid_score)
print(f'FID score for all: {fid_score}')

# compute fid on category 0
gt_dir = os.path.join(save_dir, 'gt', '0')
pred_dir = os.path.join(save_dir, 'samples', '0')
fid_score = compute_fid(gt_dir, pred_dir)
print(f'FID score for 10-20 rooms: {fid_score}')
fid_scores.append(fid_score)

# compute fid on category 1
gt_dir = os.path.join(save_dir, 'gt', '1')
pred_dir = os.path.join(save_dir, 'samples', '1')
fid_score = compute_fid(gt_dir, pred_dir)
print(f'FID score for 20-30 rooms: {fid_score}')
fid_scores.append(fid_score)

# compute fid on category 2
gt_dir = os.path.join(save_dir, 'gt', '2')
pred_dir = os.path.join(save_dir, 'samples', '2')
fid_score = compute_fid(gt_dir, pred_dir)
print(f'FID score for 30-40 rooms: {fid_score}')
fid_scores.append(fid_score)

# compute fid on category 3
gt_dir = os.path.join(save_dir, 'gt', '3')
pred_dir = os.path.join(save_dir, 'samples', '3')
fid_score = compute_fid(gt_dir, pred_dir)
print(f'FID score for over 40 rooms: {fid_score}')
fid_scores.append(fid_score)


# save the fid scores to a df, then save to csv
fid_df = pd.DataFrame({'category': ['all', '10-20', '20-30', '30-40', 'over 40'], 'fid_score': fid_scores})
fid_df.to_csv(f'{base_dir}_fid_scores.csv', index=False)


# compute iou for each category from all ious
print("Computing IoU scores")
# Load the test file
if not os.path.exists(f'{base_dir}/iou.csv'):
    print("No iou file found, please run post_process_multip.py to generate iou file")
else:
    test_df = pd.read_csv('room_num.csv')
    pairs_with_iou_df = pd.read_csv(f'{base_dir}/iou.csv', header=None, names=["source", 'IoU'])
    if len(pairs_with_iou_df) != len(test_df):
        print("The number of iou values does not match the number of samples in the test file")
        print(f"Number of iou values: {len(pairs_with_iou_df)}")
        print(f"Number of samples in the test file: {len(test_df)}")

    # Merge the two dataframes on the sample_file_name
    merged_df = pairs_with_iou_df.merge(test_df, on='source', how='left')

    mean_iou_by_category = merged_df.groupby('category')['IoU'].mean().reset_index()
    # add a row for mean iouof all categories use concat
    mean_iou_by_category = pd.concat([mean_iou_by_category, pd.DataFrame({'category': ['all'], 'IoU': [merged_df['IoU'].mean()]})])
    # change category to all, 10-20, 20-30, 30-40, over 40
    mean_iou_by_category['category'] = ['10-20', '20-30', '30-40', 'over 40', 'all']

    # merge the mean_iou_by_category with the fid_df
    test_df = fid_df.merge(mean_iou_by_category, left_on='category', right_on='category', how='left')
    test_df.to_csv(f'{base_dir}/fid_iou_scores.csv', index=False)
    print(test_df)
