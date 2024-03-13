import os

def split_data(image_list, label_list, train_ratio=0.8):
    # 데이터셋 길이
    total_length = len(image_list)
    train_length = int(total_length * train_ratio)
    
    # 훈련 데이터와 검증 데이터로 분할
    train_images = image_list[:train_length]
    train_labels = label_list[:train_length]
    val_images = image_list[train_length:]
    val_labels = label_list[train_length:]

    # image 은 prefix 로 images, label 은 prefix 로 labels 를 추가
    train_images = ['images/'+x for x in train_images]
    train_labels = ['labels/'+x for x in train_labels]
    val_images = ['images/'+x for x in val_images]
    val_labels = ['labels/'+x for x in val_labels]
    
    return (train_images, train_labels, val_images, val_labels)

def check(name1, name2):
    token = name1.split('/')[-1].split('.')[0]

    # 대응되는지 확인
    assert token in name2
    return

if __name__ == '__main__':
    output_directory = '../data/list/endoscope'
    image_directory = '../data/endoscope/images'
    label_directory = '../data/endoscope/labels'

    # 파일 목록을 가져오고 정렬
    image_list = sorted(os.listdir(image_directory))
    label_list = sorted([x for x in os.listdir(label_directory) if 'labelIds' in x])

    assert len(image_list) == len(label_list), "Image and Label counts do not match."

    # 데이터 분할
    train_images, train_labels, val_images, val_labels = split_data(image_list, label_list)

    # 파일로 쓰기
    with open(os.path.join(output_directory, 'train.lst'), 'w') as f_train, \
         open(os.path.join(output_directory, 'val.lst'), 'w') as f_val:
        
        for img, lbl in zip(train_images, train_labels):
            # 동일한 쌍인지 한 번 더 확인
            check(img, lbl)
            f_train.write(f"{img}\t{lbl}\n")
        
        for img, lbl in zip(val_images, val_labels):
            # 동일한 쌍인지 한 번 더 확인
            check(img, lbl)
            f_val.write(f"{img}\t{lbl}\n")

    print("Data split and written to .lst files successfully.")
