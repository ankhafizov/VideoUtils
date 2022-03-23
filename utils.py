
import numpy as np
import cv2
from tqdm import tqdm

import os


def downsize(scale, input_video_path, output_video_filename, fps):
    """
    scale > 1
    """

    video_stream = cv2.VideoCapture(input_video_path)

    video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 300)

    frame_width = int(video_stream.get(3)) // scale
    frame_height = int(video_stream.get(4)) // scale
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, [frame_width, frame_height])
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video_stream.release()
    out.release()
    cv2.destroyAllWindows()


def flip(axis, input_video_path, output_video_filename, fps):
    """
    flip frames
    """

    video_stream = cv2.VideoCapture(input_video_path)
    video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 300)

    frame_width, frame_height = int(video_stream.get(3)), int(video_stream.get(4))
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = np.flip(frame, axis=axis)

        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video_stream.release()
    out.release()
    cv2.destroyAllWindows()


def reverse(input_video_path, output_video_filename, fps):
    """
    flip frames
    """

    video_stream = cv2.VideoCapture(input_video_path)
    frame_width, frame_height = int(video_stream.get(3)), int(video_stream.get(4))

    frames = []
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frames.append(frame)

    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    frames.reverse()

    for frame in frames:
        out.write(frame)
        cv2.imshow('frame', frame)

    video_stream.release()
    out.release()
    cv2.destroyAllWindows()


def h_merge(input_video_paths, output_video_filename, fps):
    """
    input_video_paths - list
    """

    video_streams = [cv2.VideoCapture(p) for p in input_video_paths]

    widths = np.array([vs.get(3) for vs in video_streams])
    height = np.array([vs.get(4) for vs in video_streams])

    assert(np.all(widths == widths[0]))
    assert(np.all(height == height[0]))

    frame_width, frame_height = int(np.sum(widths)), int(height[0])
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    while video_streams[0].isOpened():
        rets, frames = np.transpose([vs.read() for vs in video_streams])
        if not np.all(rets):
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = np.hstack(frames)

        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    [vs.release() for vs in video_streams]
    out.release()
    cv2.destroyAllWindows()


def add_empty_frames(where, secs, input_video_path, output_video_filename, new_fps=False):
    """
    where - "start" or "end"
    """

    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS) if not new_fps else new_fps
    frame_width, frame_height = int(video_stream.get(3)), int(video_stream.get(4))
    generate_black_frame = lambda: np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    frames = []
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frames.append(frame)
    count_frames_to_add = int(fps * secs)
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    if where == "start":
        [out.write(generate_black_frame()) for _ in tqdm(range(count_frames_to_add))]
        [out.write(f) for f in tqdm(frames)]
    elif where == "end":
        [out.write(f) for f in tqdm(frames)]
        [out.write(generate_black_frame()) for _ in tqdm(range(count_frames_to_add))]
    
    video_stream.release()
    out.release()


def trim(start_secs, stop_secs, input_video_path, output_video_filename, new_fps=False):
    """
    where - "start" or "end"
    """

    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = int(video_stream.get(3)), int(video_stream.get(4))
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps if not new_fps else new_fps, (frame_width, frame_height))

    start_N = int(fps * start_secs)
    stop_N = int(fps * stop_secs) if stop_secs is not np.inf else np.inf
    i = 0

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret or i == stop_N:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if i >= start_N:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            out.write(frame)
        i += 1

    cv2.destroyAllWindows()
    
    video_stream.release()
    out.release()


def change_duration(duration_secs, input_video_path, output_video_filename, new_fps=False):
    """
    увеличить (или уменьшить) в несколько раз путем заикливания до определенного количества секунд duration_secs
    """

    video_stream = cv2.VideoCapture(input_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS) if not new_fps else new_fps
    frame_width, frame_height = int(video_stream.get(3)), int(video_stream.get(4))

    frames = []
    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frames.append(frame)
    
    current_duration = fps * len(frames)
    assert(current_duration > duration_secs)

    N_current = len(frames)
    N_dest = int(duration_secs * fps)

    frames = frames * (N_dest // N_current) + frames[:N_dest % N_current]

    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    [out.write(f) for f in tqdm(frames)]
    
    video_stream.release()
    out.release()

if __name__ == "__main__":
  