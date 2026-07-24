from utils.video_loader import VideoLoader

loader = VideoLoader()

path = input("Enter video path: ")

info = loader.open(path)

if info is None:
    print("Unable to open video.")
else:
    print("\nVideo Information")
    print("------------------")
    print(f"Resolution : {info['width']} x {info['height']}")
    print(f"FPS        : {info['fps']:.2f}")
    print(f"Frames     : {info['frames']}")
    print(f"Duration   : {info['duration']:.2f} sec")

loader.release()