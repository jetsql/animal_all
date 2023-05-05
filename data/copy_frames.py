import os
import shutil
from tqdm import tqdm

# video_path = '/media/perry/wd8t/paper_data/original_single_process/FPS_1_data/history_record/8_sec_2/cut_videos'
# location_a='/media/perry/wd8t/paper_data/original_single_process/FPS_1_data/history_record/8_sec_2/labelframes'
# location_b='/media/perry/wd8t/paper_data/original_single_process/FPS_1_data/history_record/non_cut/8_sec/labelframes'
# os.chdir(video_path)
# print(os.getcwd())
# def copy(video_id_A,file_name_A,video_id_B,file_name_B):
#     shutil.copyfile(os.path.join(location_a,video_id_A+'_'+format(file_name_A)+'.jpg'),
#                         os.path.join(location_b, video_id_B+'_'+format(file_name_B)+'.jpg'))
# video_ids = [video_id[:-4] for video_id in os.listdir(video_path)]
# for video_id in tqdm(video_ids):
#     #檔名_1，_2要改
#     #ffmpeg.sh要改
#     video_id_A=video_id+"_2"
#     video_id_B=video_id+"_1"
#     # for img_id in range(2*fps+1, (seconds-2)*30, fps):
#     copy(video_id_A,'00001',video_id_B,'00005')
#     copy(video_id_A,'00002',video_id_B,'00006')
#     copy(video_id_A,'00003',video_id_B,'00007')
#     copy(video_id_A,'00004',video_id_B,'00008')


print("===load===")
print(os.getcwd())
os.chdir('/home/perry/Desktop/code/mmaction2_test/data')
print(os.getcwd())
video_path = './ava/videos'
labelframes_path = './ava/labelframes'
rawframes_path = './ava/rawframes'
video_ids = [video_id[:-4] for video_id in os.listdir(video_path)]
def copy(video_id,file_name_A,file_name_B):
    shutil.copyfile(os.path.join(rawframes_path, video_id, 'img_'+format(file_name_A)+'.jpg'),
                        os.path.join(labelframes_path, video_id+'_'+format(file_name_B)+'.jpg'))
for video_id in tqdm(video_ids):
    #檔名_1，_2要改
    #ffmpeg.sh要改
    video_id=video_id+"_1"
    # for img_id in range(2*fps+1, (seconds-2)*30, fps):
    copy(video_id,'00001','00001')
    copy(video_id,'00031','00002')
    copy(video_id,'00061','00003')
    copy(video_id,'00091','00004')
    copy(video_id,'00121','00005')
    copy(video_id,'00151','00006')
    copy(video_id,'00181','00007')
    copy(video_id,'00211','00008')