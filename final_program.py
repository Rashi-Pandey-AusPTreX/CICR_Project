
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Loop for camera indices and backends to find a working camera
cap = None
for i in range(5):
   for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        print("Trying camera:", i, "Backend:", backend)
        test = cv2.VideoCapture(i, backend)
        if test.isOpened():
            ret, frame = test.read()
            if ret:
                cap = test
                print("Camera working at index:", i)
                print("Backend:", backend)
                break
        test.release()

if cap is None:
    print("No working camera found!")
    exit()

print("AI Object Tracking & Counting Started!")

window_name = "AI Object Tracking & Counting"

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Detect and Track
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    output = frame.copy()

    counts = {}

    # Zone Monitoring
    height, width = frame.shape[:2]

    zone_x1 = int(width * 0.3)
    zone_y1 = int(height * 0.25)
    zone_x2 = int(width * 0.7)
    zone_y2 = int(height * 0.75)

    zone_count = 0

    # Draw Zone
    cv2.rectangle(
        output,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        "MONITORING ZONE",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    if results[0].boxes is not None:

        for box in results[0].boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class information
            cls = int(box.cls[0])
            name = model.names[cls]

            # Confidence Level
            confidence = float(box.conf[0]) * 100

            # Count objects
            counts[name] = counts.get(name, 0) + 1

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Find centre of object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if object is inside zone
            if (zone_x1 <= center_x <= zone_x2 and
                zone_y1 <= center_y <= zone_y2):

                zone_count += 1
                box_color = (0, 0, 255)

            else:
                box_color = (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Label
            label = f"{name} | ID:{track_id} | Conf:{confidence:.1f}%"

            # Label Background
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 35)),
                (x1 + text_size[0] + 10, y1),
                (0, 0, 0),
                -1
            )

            # Draw Label
            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    # Display Count Information
    y = 30

    for obj_name, count in counts.items():

        cv2.putText(
            output,
            f"{obj_name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        y += 35

    # Total count
    total_count = sum(counts.values())

    cv2.putText(
        output,
        f"Total Count: {total_count}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y += 40

    # Display Zone Count
    cv2.putText(
        output,
        f"Zone Count: {zone_count}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Show frame
    cv2.imshow(window_name, output)

    # Keyboard Input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Program terminated by user.")
        break
    elif key == ord('Q'):
        print("Program terminated by user.")
        break

    # Window Close Detection
    try:
        if cv2.getWindowProperty(
            window_name,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            print("Window closed.")
            break
    except:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Program ended successfully.")
