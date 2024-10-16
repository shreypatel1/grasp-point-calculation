import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from imutils.video import VideoStream, FPS

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--source", type=str, default="0", help="source")
ap.add_argument("--weights", type=str, default="runs/segment/300epochs/weights/best.pt", help="weights")
ap.add_argument("--conf", type=float, default=0.90, help="confidence threshold")
ap.add_argument("--save", action='store_true', help="save")
ap.add_argument("--show", action='store_true', help="show")
args = ap.parse_args()

# Initialize YOLO model
model = YOLO(args.weights)

# Initialize the video stream and FPS counter
vs = VideoStream(src=0).start()  # Start camera
fps = FPS().start()  # Start FPS calculation

# Loop over the frames from the video stream
while True:
    frame = vs.read()
    # frame = cv2.resize(frame, (640, 416))

    # Run the YOLO model on the frame
    results = model(frame, conf=args.conf, stream=True)


    for r in results:
        if r.masks is not None:  # Ensure there are masks in the results
            # Extract the mask and convert it to a NumPy array
            r.bbox = r.bbox[0]
            mask = r.masks.data[0].cpu().numpy()
            # Resize the mask to match the frame size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            # Convert mask to a binary mask (threshold at 0.5)
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255  # Convert to 255 for display

            # Optionally apply the mask to the original frame and show the result
            # masked_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
            # cv2.imshow("Masked Frame", masked_frame)

            # Highlight the mask region in the original frame
            frame = cv2.addWeighted(frame, 1, cv2.merge((binary_mask, binary_mask, binary_mask)), 0.5, 0)

            # If no results, still show the mask and masked frame with zeros
            # cv2.imshow("Mask", np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8))
            # cv2.imshow("Masked Frame", np.zeros_like(frame))

    cv2.imshow("Frame", frame)
    print(frame.shape[1], frame.shape[0])

    # Wait for the `q` key to be pressed to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()
    # print("Avg FPS: ", fps.fps())

# Stop the FPS counter and clean up
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()
cv2.destroyAllWindows()
