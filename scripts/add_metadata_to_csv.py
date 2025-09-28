import pandas as pd
import cv2
import concurrent.futures
from tqdm import tqdm
import argparse

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return width, height, fps, frame_count, duration

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2] # h w c
    fps = 0
    frame_count = 1
    duration = 0
    return width, height, fps, frame_count, duration

def get_data_info(data_path: str):
    if data_path.lower().endswith(('.mp4', '.mov', '.m4v')):
        return get_video_dimensions(data_path)
    elif data_path.lower().endswith(('.jpeg', '.jpg', '.png')):
        return get_image_dimensions(data_path)
    else:
        raise NotImplementedError(data_path)

def process(df, data_column):
    data_paths = df[data_column].tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(get_data_info, data_paths), total=len(data_paths)))
    
    df["width"], df["height"], df["fps"], df["frame_count"], df["duration"] = zip(*results)
    df = df.dropna(subset=["width", "height", "fps", "frame_count", "duration"]) # drop None

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--csv_path", type=str, required=True)
    parser.add_argument("-o", "--save_path", type=str, default=None)
    parser.add_argument("--data_column", type=str, default="video_path")

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.csv_path

    print(f'src: {args.csv_path}')
    print(f'dst: {args.save_path}')
    df = pd.read_csv(args.csv_path)
    n = len(df)
    df = process(df, args.data_column)
    print(f'{args.save_path}: {n}->{len(df)}')
    df.to_csv(args.save_path, index=False)