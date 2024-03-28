import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter

# CSV 파일 로드
csv_file_path = '../output/endoscope/pidnet_small_endoscope/2024-03-28-09-54/results.csv'
df = pd.read_csv(csv_file_path)

# dir 명 추출
subdir = ''
if 'small' in csv_file_path:
    subdir = 'small_' + csv_file_path.split('/')[-2]
elif 'medium' in csv_file_path:
    subdir = 'medium_' + csv_file_path.split('/')[-2]
elif 'large' in csv_file_path:
    subdir = 'large_' + csv_file_path.split('/')[-2]

# 로깅할 디렉토리 설정
log_dir = os.path.join('tensorboard_logs', subdir)
os.makedirs(log_dir, exist_ok=True)

# TensorBoard SummaryWriter 초기화
writer = SummaryWriter(log_dir)

# 각 epoch에 대해 각 필드의 값을 로그
for index, row in df.iterrows():
    epoch = row['epoch']
    for field in df.columns:
        if field != 'epoch':
            # add_scalar 함수를 사용하여 로그
            writer.add_scalar(f'mIoU/{field}', row[field], epoch)

# 로깅 완료 후 SummaryWriter 닫기
writer.close()

# TensorBoard 시작 명령어 (터미널에서 실행해야 함)
print(f'TensorBoard를 시작하려면 터미널에서 다음 명령어를 실행하세요: tensorboard --logdir={log_dir}')
