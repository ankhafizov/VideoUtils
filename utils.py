
import numpy as np
import cv2


def downsize(scale, input_video_path, output_video_filename, fps):
    """
    scale > 1
    """

    video_stream = cv2.VideoCapture(input_video_path)

    video_stream.set(cv2.CAP_PROP_BUFFERSIZE, 300)

    frame_width = int(video_stream.get(3)) // scale
    frame_height = int(video_stream.get(4)) // scale
    out = cv2.VideoWriter(output_video_filename,
                          cv2.VideoWriter_fourcc(*'XVID'),
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
                          cv2.VideoWriter_fourcc(*'XVID'),
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
                          cv2.VideoWriter_fourcc(*'XVID'),
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
                          cv2.VideoWriter_fourcc(*'XVID'),
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


if __name__ == "__main__":
    fps = 25
    reverse("plate-recognition-server_1.avi",
            "plate-recognition-server.avi", fps)
