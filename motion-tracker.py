import cv2
import numpy as np
import os

# ---------- SETTINGS ----------
VIDEO_PATH = "NeckDeformationClip.mp4"   # مسیر ویدئوی گردن
OUTPUT_PATH = "neck_motion_output_tracker.mp4"   # مسیر ذخیره خروجی
DISPLAY_SCALE = 1.0                      # مقیاس نمایش (1 = اندازه واقعی)
SAVE_VIDEO = True                        # اگر True باشه خروجی ذخیره میشه
# ------------------------------

# باز کردن ویدئو
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError("❌ ویدئو پیدا نشد یا باز نشد!")

# گرفتن اطلاعات ویدئو
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# تنظیم خروجی ویدئو
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# گرفتن اولین فریم و تبدیل به grayscale
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("❌ خطا در خواندن فریم اول ویدئو")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_count = 0

print("🎥 Optical Flow tracking started... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ پردازش تمام شد.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # محاسبه Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # جدا کردن مؤلفه‌های حرکت (x, y)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ساخت تصویر HSV برای نمایش رنگی جهت و شدت حرکت
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    # Hue = زاویه حرکت، Value = شدت حرکت
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    motion_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # نمایش کنار هم برای مقایسه
    combined = np.hstack((frame, motion_color))

    # تغییر اندازه برای نمایش بهتر
    if DISPLAY_SCALE != 1.0:
        combined = cv2.resize(combined, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    cv2.imshow("Neck Optical Flow (ESC to exit)", combined)

    # ذخیره در فایل خروجی
    if SAVE_VIDEO:
        out.write(motion_color)

    # بستن با دکمه ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_gray = gray.copy()
    frame_count += 1

cap.release()
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()

print(f"💾 ویدئو ذخیره شد به عنوان: {OUTPUT_PATH}")
print(f"📊 کل فریم‌های پردازش شده: {frame_count}")
