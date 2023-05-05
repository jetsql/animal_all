import os
import shutil
from tqdm import tqdm
#檔名_1，_2要改 ffmpeg.sh要改
# 0/4/_1
# 4/4/_2
# 8/4/_3
# 12/4/_4
# 16/4/_5


start = 0
seconds = 4
#############

print("===load===")
print(os.getcwd())
os.chdir('/home/perry/Desktop/code/copy_code/mmaction2_animal/data')
print(os.getcwd())
video_path = './ava/videos'
labelframes_path = './ava/labelframes'
rawframes_path = './ava/rawframes'
cut_videos_sh_path = './cut_videos.sh'
 
# if os.path.exists(labelframes_path):
#     shutil.rmtree(labelframes_path)
# if os.path.exists(rawframes_path):
#     shutil.rmtree(rawframes_path)
 
fps = 30
raw_frames = seconds*fps
 
with open(cut_videos_sh_path, 'r') as f:
    sh = f.read()
sh = sh.replace(sh[sh.find('    ffmpeg'):], f'    ffmpeg -ss {start} -t {seconds} -i "${{video}}" -r 30 -strict experimental "${{out_name}}"\n  fi\ndone\n')
with open(cut_videos_sh_path, 'w') as f:
    f.write(sh)
# 902打到1798
os.system('bash cut_videos.sh')
os.system('bash extract_rgb_frames_ffmpeg.sh')
os.makedirs(labelframes_path, exist_ok=True)
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

    # copy(video_id,'00121','00005')
    # copy(video_id,'00151','00006')
    # copy(video_id,'00181','00007')
    # copy(video_id,'00211','00008')

    # copy(video_id,'00241','00009')
    # copy(video_id,'00271','00010')
    # copy(video_id,'00301','00011')
    # copy(video_id,'00331','00012')

    # copy(video_id,'00361','00013')
    # copy(video_id,'00391','00014')
    # copy(video_id,'00421','00015')
    # copy(video_id,'00451','00016')

    # copy(video_id,'00481','00017')
    # copy(video_id,'00511','00018')
    # copy(video_id,'00541','00019')
    # copy(video_id,'00571','00020')

    # copy(video_id,'00601','00021')
    # copy(video_id,'00631','00022')
    # copy(video_id,'00661','00023')
    # copy(video_id,'00691','00024')

    # copy(video_id,'00721','00025')
    # copy(video_id,'00751','00026')
    # copy(video_id,'00781','00027')
    # copy(video_id,'00811','00028')

    #原始程式
    # for img_id in range(1, 120):
    #     shutil.copyfile(os.path.join(rawframes_path, video_id, 'img_'+format(img_id, '%05d')+'.jpg'),
    #                     os.path.join(labelframes_path, video_id+'_'+format(start+img_id//30, '%05d')+'.jpg'))